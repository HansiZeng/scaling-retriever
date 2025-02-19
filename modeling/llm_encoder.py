import torch 
from transformers import T5ForConditionalGeneration, LlamaForCausalLM, BertForMaskedLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import ujson

from .losses.regulariaztion import init_regularizer
from .bidirectional_llama import LlamaBiForMNTP, LlamaBiModel
from .bidrectional_qwen2 import Qwen2BiForMNTP, Qwen2BiModel


class LLM2Retriever(torch.nn.Module):
    _tied_weights_keys = None
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # by default, we use NCE loss 
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.reg_loss = init_regularizer("FLOPS")
        
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            self.local_rank = torch.distributed.get_rank()
        
    def encode(self, **inputs):
        raise NotImplementedError
    
    def gather(self, tensor):
        dtensor = tensor.detach()
        gather_list = [torch.zeros_like(dtensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, dtensor)
        gather_list[self.local_rank] = tensor 
        
        return torch.cat(gather_list, 0)
    
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        rank_loss = self.loss_fn(logits, labels)
        
        query_reg_loss = self.reg_loss(query_reps)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }    
    
    def doc_encode(self, **inputs):
        return self.encode(**inputs)
    
    def query_encode(self, **inputs):
        return self.encode(**inputs)
    
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    @classmethod 
    def build(cls, model_name_or_path, args, config=None):
        if config is not None:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            model_config.update(config)
            print("to modify model config: ", config)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, config=model_config)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if args.lora:
            lora_config = LoraConfig(
                    base_model_name_or_path=args.model_name_or_path,
                    task_type=None,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=cls.TARGET_MODULES,
                    inference_mode=False,
                    modules_to_save=args.lora_modules_to_save
                )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(lora_model) 
            lora_model.print_trainable_parameters()
        else:
            model = cls(base_model)
            
        return model
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, 
                                                   lora_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=is_trainable)
            if merge_peft:
                lora_model = lora_model.merge_and_unload()
            else:
                lora_model.print_trainable_parameters()
            model = cls(lora_model)
        else:
            model = cls(base_model)
        
        return model
            
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)
   
        
class T5Splade(LLM2Retriever):
    TRANSFORMER_CLS = T5ForConditionalGeneration
    TARGET_MODULES = ["q", "v", "o", "k", "wi_0", "wi_1", "wo"]
    
    def __init__(self, base_model):
        super().__init__(base_model)
        self.vocab_size = self.base_model.config.vocab_size
        
    def encode(self, **inputs):
        assert "decoder_input_ids" in inputs, inputs.keys()
        seq_reps = self.base_model(**inputs, return_dict=True).logits #[bz, seq_length, dim]
        if self.base_model.config.d_model >=2048:
            seq_reps *= self.base_model.model_dim**-0.25

        reps, _ = torch.max(torch.log(1 + torch.relu(seq_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1) #[bz, vocab_size]
        
        return reps


class DecoderOnlyBiSplade(LLM2Retriever):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.vocab_size = self.base_model.config.vocab_size
        
    def rerank_forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"])
        doc_reps = self.encode(**inputs["tokenized_docs"])
        logits = (query_reps * doc_reps).sum(dim=-1)
        return logits
    
    def encode(self, **inputs):
        seq_reps = self.base_model(**inputs, return_dict=True).logits #[bz, seq_length, dim]
        seq_reps *= self.base_model.config.hidden_size**-0.25
        
        # reps, _ = torch.max(torch.log(1 + torch.relu(seq_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1) #[bz, vocab_size] 
        
        ## we try efficient encode to see whether it can save memory 
        reps = torch.log(torch.relu(torch.max(seq_reps + ( 1 - inputs["attention_mask"].unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
        
        
        return reps
    

class LlamaBiSplade(DecoderOnlyBiSplade):
    TRANSFORMER_CLS = LlamaBiForMNTP
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
    
class Qwen2BiSplade(DecoderOnlyBiSplade):
    TRANSFORMER_CLS = Qwen2BiForMNTP
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
LlamaBiSpladeForNCE = LlamaBiSplade 
Qwen2BiSpladeForNCE = Qwen2BiSplade


class LlamaBiSpladeForMarginMSE(LlamaBiSplade):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
        

class LlamaBiSpladeForNCE_KLDiv(LlamaBiSplade):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.nce_loss = torch.nn.CrossEntropyLoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_loss = init_regularizer("FLOPS")
        
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        teacher_scores = inputs["teacher_scores"] # [n_query, 1 + num_negs]
        teacher_idxes = inputs["teacher_idxes"] # [n_query, 1 + num_negs]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
            # the following logits is cross device.
            # The kl_div loss only consider the logits in the same device
            # as teacher_scores is in the same device
            teacher_idxes = teacher_idxes.view(-1) + self.local_rank * n_context 
            
            # original `query_idexes` is wrong
            # query_idxes = torch.LongTensor([self.local_rank] * n_context).to(labels.device)
            query_idxes = torch.LongTensor(torch.repeat_interleave(torch.arange(n_query), n_context // n_query)) + \
                            self.local_rank * n_query
            query_idxes = query_idxes.to(labels.device)
        
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        nce_loss = self.loss_fn(logits, labels)
        
        kl_logits = logits[query_idxes, teacher_idxes].view(teacher_scores.size())
        log_probs = torch.nn.functional.log_softmax(kl_logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        kl_loss = self.kldiv_loss(log_probs, teacher_log_probs)
        
        rank_loss = (nce_loss + kl_loss) / 2.
        
        query_reg_loss = self.reg_loss(query_reps)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        # rank_loss is used for backward 
        # we also inspect kldiv and nce losses.
        return {
            "rank": rank_loss, 
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
        
        
class LlamaBiSpladeForKLDiv(LlamaBiSplade):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_loss = init_regularizer("FLOPS")
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_queries"])
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        teacher_scores = inputs["teacher_scores"] #[bz, 1 + num_negs]
        
        context_reps = context_reps.view(teacher_scores.size(0),
                                         teacher_scores.size(1),
                                         context_reps.size(-1))
        logits = (query_rep.unsqueeze(1) * context_reps).sum(dim=-1) #[bz, 1 + num_negs] 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        rank_loss = self.rank_loss(log_probs, teacher_log_probs)
        
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        return {
            "rank": rank_loss, 
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }


class Qwen2BiSpladeForMarginMSE(Qwen2BiSplade):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
        

class T5SpladeForMarginMSE(T5Splade):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }


class DecoderOnlyBiDense(LLM2Retriever):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.hidden_size = self.base_model.config.hidden_size
        self.T = T
        
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        
        rank_loss = self.loss_fn(logits / self.T, labels)
        
        return rank_loss
    
    def _debug_forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        rank_loss = self.loss_fn(logits, labels)
        
        return rank_loss, logits, labels
    
    def rerank_forward(self, **inputs):
        # it is just for evaluation
        query_reps = self.encode(**inputs["tokenized_queries"])
        doc_reps = self.encode(**inputs["tokenized_docs"])
        logits = (query_reps * doc_reps).sum(dim=-1)
        return logits
    
    def encode(self, **inputs):
        # since we do left padding, and add the cls_token_id to the last position.
        # but we make sure that it is correctly implemented 
        #seq_reps = self.base_model(**inputs, return_dict=True).last_hidden_state 
        #reps = seq_reps[:, -1] #[bz, dim]
        
        # we do average embedding
        # padding_size is from left 
        seq_lengths = inputs["attention_mask"].sum(dim=-1)
        seq_reps = self.base_model(**inputs, return_dict=True).last_hidden_state 
        seq_reps = torch.nn.functional.normalize(seq_reps, p=2, dim=-1)
        reps = torch.stack(
                [
                    seq_reps[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
        )
        
        return reps 
    
    @classmethod 
    def build(cls, model_name_or_path, args, config=None):
        if config is not None:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            model_config.update(config)
            print("to modify model config: ", config)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, config=model_config)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if args.lora:
            lora_config = LoraConfig(
                    base_model_name_or_path=args.model_name_or_path,
                    task_type=None,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=cls.TARGET_MODULES,
                    inference_mode=False,
                    modules_to_save=args.lora_modules_to_save
                )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(lora_model, T=args.T) 
            lora_model.print_trainable_parameters()
        else:
            model = cls(base_model, T=args.T)
            
        return model
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False,
             T=0.01):
        if lora_name_or_path is not None:
            # It is hacky here, but we need to check wether the lora_name_or_path is with the expected format
            from safetensors.torch import load_file
            import os
            if os.path.exists(os.path.join(lora_name_or_path, "adapter_model.safetensors")):
                tmp_state_dict = load_file(os.path.join(lora_name_or_path, "adapter_model.safetensors"))
            elif os.path.exists(os.path.join(lora_name_or_path, "adapter_model.bin")):
                tmp_state_dict = torch.load(os.path.join(lora_name_or_path, "adapter_model.bin"))
            assert "base_model.model.model.layers" not in list(tmp_state_dict.keys())[0]
            assert "base_model.model.layers" in list(tmp_state_dict.keys())[0]
            tmp_state_dict = None
                
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, 
                                                   lora_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=is_trainable)
            if merge_peft:
                lora_model = lora_model.merge_and_unload()
            model = cls(lora_model, T=T)
            
            # we also check lorr_config here 
            assert lora_config.auto_mapping["base_model_class"] == cls.TRANSFORMER_CLS.__name__, (
                lora_config.auto_mapping["base_model_class"], cls.TRANSFORMER_CLS.__name__
            )
            if not merge_peft:
                lora_model.print_trainable_parameters()
        else:
            model = cls(base_model, T=T)
        
        return model


class LlamaBiDense(DecoderOnlyBiDense):
    TRANSFORMER_CLS = LlamaBiModel
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    

class Qwen2BiDense(DecoderOnlyBiDense):
    TRANSFORMER_CLS = Qwen2BiModel
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
LlamaBiDenseForNCE = LlamaBiDense
Qwen2BiDenseForNCE = Qwen2BiDense


class LlamaBiDenseForMarginMSE(LlamaBiDense):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        self.T = T
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, hdim]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin / self.T, teacher_margin)

        return rank_loss
    

class LlamaBiDenseForKLDiv(LlamaBiDense):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.rank_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.T = T
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_queries"])
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        teacher_scores = inputs["teacher_scores"] #[bz, 1 + num_negs]
        
        context_reps = context_reps.view(teacher_scores.size(0),
                                         teacher_scores.size(1),
                                         context_reps.size(-1))
        logits = (query_rep.unsqueeze(1) * context_reps).sum(dim=-1) / self.T #[bz, 1 + num_negs] 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        rank_loss = self.rank_loss(log_probs, teacher_log_probs)
        
        return rank_loss
    

class LlamaBiDenseForNCE_KLDiv(LlamaBiDense):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.nce_loss = torch.nn.CrossEntropyLoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.T = T
        
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        teacher_scores = inputs["teacher_scores"] # [n_query, 1 + num_negs]
        teacher_idxes = inputs["teacher_idxes"] # [n_query, 1 + num_negs]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        # n_context / n_query == 1 + num_negs
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
            # the following logits is cross device.
            # The kl_div loss only consider the logits in the same device
            # as teacher_scores is in the same device
            teacher_idxes = teacher_idxes.view(-1) + self.local_rank * n_context 
            
            # original `query_idexes` is wrong
            # query_idxes = torch.LongTensor([self.local_rank] * n_context).to(labels.device)
            query_idxes = torch.LongTensor(torch.repeat_interleave(torch.arange(n_query), n_context // n_query)) + \
                            self.local_rank * n_query
            query_idxes = query_idxes.to(labels.device)
            
        
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        nce_loss = self.loss_fn(logits / self.T, labels)
        
        kl_logits = logits[query_idxes, teacher_idxes].view(teacher_scores.size()) / self.T
        log_probs = torch.nn.functional.log_softmax(kl_logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        kl_loss = self.kldiv_loss(log_probs, teacher_log_probs)
        
        rank_loss = (nce_loss + kl_loss) / 2.
        
        # rank_loss is used for backward 
        # we also inspect kldiv and nce losses.
        return {
            "loss": rank_loss, 
            "nce": nce_loss.detach(),
            "kldiv": kl_loss.detach(),
        }


class Qwen2BiDenseForMarginMSE(Qwen2BiDense):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        self.T = T
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, hdim]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin / self.T, teacher_margin)

        return rank_loss
    
