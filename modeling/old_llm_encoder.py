class LlamaBiDense(LLM2Retriever):
    TRANSFORMER_CLS = LlamaBiModel
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
    def __init__(self, base_model):
        super().__init__(base_model)
        self.hidden_size = self.base_model.config.hidden_size
        
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
        seq_reps *= self.base_model.config.hidden_size**-0.25
        reps = torch.stack(
                [
                    seq_reps[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
        )
        
        return reps 
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False):
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
            model = cls(lora_model)
            
            # we also check lorr_config here 
            assert lora_config.auto_mapping["base_model_class"] == cls.TRANSFORMER_CLS.__name__, (
                lora_config.auto_mapping["base_model_class"], cls.TRANSFORMER_CLS.__name__
            )
            if not merge_peft:
                lora_model.print_trainable_parameters()
        else:
            model = cls(base_model)
        
        return model
    

class LlamaBiSpladeOld(LLM2Retriever):
    TRANSFORMER_CLS = LlamaBiForMNTP
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
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


class LlamaBiDenseOld(LLM2Retriever):
    TRANSFORMER_CLS = LlamaBiModel
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    
    def __init__(self, base_model):
        super().__init__(base_model)
        self.hidden_size = self.base_model.config.hidden_size
        
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
        seq_reps *= self.base_model.config.hidden_size**-0.25
        reps = torch.stack(
                [
                    seq_reps[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
        )
        
        return reps 
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False):
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
            model = cls(lora_model)
            
            # we also check lorr_config here 
            assert lora_config.auto_mapping["base_model_class"] == cls.TRANSFORMER_CLS.__name__, (
                lora_config.auto_mapping["base_model_class"], cls.TRANSFORMER_CLS.__name__
            )
            if not merge_peft:
                lora_model.print_trainable_parameters()
        else:
            model = cls(base_model)
        
        return model



class LlamaBiHybridRetrieverForNCE(LLM2Retriever):
    # combine the sparse (splade) and dense retrievers together.
    TRANSFORMER_CLS = LlamaBiForMNTP
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    cls_token_id = 128009
    
    def __init__(self, base_model):
        super().__init__(base_model)
        self.vocab_size = self.base_model.config.vocab_size
        self.hidden_size = self.base_model.config.hidden_size
        
    def lexical_encode(self, **inputs):
        # this is only used for inference
        # since we do left padding, and add the cls_token_id to the last position.
        # but we make sure that it is correctly implemented 
        
        model_output = self.base_model(**inputs, return_dict=True, output_hidden_states=True)
        lexical_reps = model_output.logits * self.hidden_size**-0.25 #[bz, seq_length-1, dim]
        lexical_reps, _ = torch.max(torch.log(1 + torch.relu(lexical_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1)
        
        return lexical_reps

    def dense_encode(self, **inputs):
        model_output = self.base_model(**inputs, return_dict=True, output_hidden_states=True)
        dense_reps = self._get_dense_reps(model_output, inputs["attention_mask"])
        
        return dense_reps

    def _get_dense_reps(self, model_output, attention_mask):
        seq_reps = model_output.hidden_states[-1] 
        seq_reps = seq_reps * self.base_model.config.hidden_size**-0.25
        seq_lengths = attention_mask.sum(dim=-1)
        reps = torch.stack(
                [
                    seq_reps[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
        )
        return reps 
    
    def encode(self, **inputs):
        model_output = self.base_model(**inputs, return_dict=True, output_hidden_states=True)
        
        lexical_reps = model_output.logits * self.hidden_size**-0.25 #[bz, seq_length, dim]
        lexical_reps, _ = torch.max(torch.log(1 + torch.relu(lexical_reps)) * inputs["attention_mask"].unsqueeze(-1), dim=1)
        
        # dense 
        dense_reps = self._get_dense_reps(model_output, inputs["attention_mask"])
        
        return lexical_reps, dense_reps
    
    def _gather_query_context(self, query_reps, context_reps):
        query_reps = self.gather(query_reps)
        context_reps = self.gather(context_reps)
        
    
    def forward(self, **inputs):
        lex_query_reps, dense_query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        lex_context_reps, dense_context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = dense_query_reps.size(0)
        n_context = dense_context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            # lex 
            lex_query_reps = self.gather(lex_query_reps)
            lex_context_reps = self.gather(lex_context_reps)
            # dense
            dense_query_reps = self.gather(dense_query_reps)
            dense_context_reps = self.gather(dense_context_reps)
            
            # labels
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
        
        lex_logits = torch.matmul(lex_query_reps, lex_context_reps.transpose(1,0))
        dense_logits = torch.matmul(dense_query_reps, dense_context_reps.transpose(1,0))
        hybrid_logits = (lex_logits + dense_logits) / 2.
        
        query_reg_loss = self.reg_loss(lex_query_reps)
        doc_reg_loss = self.reg_loss(lex_context_reps) 
        
        return {
            "lex_rank": self.loss_fn(lex_logits, labels),
            "dense_rank": self.loss_fn(dense_logits, labels),
            "hybrid_rank": self.loss_fn(hybrid_logits, labels),
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
        
    def rerank_forward(self, **inputs):
        lex_query_reps, dense_query_reps = self.encode(**inputs["tokenized_queries"])
        lex_doc_reps, dense_doc_reps = self.encode(**inputs["tokenized_docs"])
        logits = (lex_query_reps * lex_doc_reps).sum(dim=-1) + (dense_query_reps * dense_doc_reps).sum(dim=-1)
        
        return logits


class LlamaBiHybridRetrieverForMarginMSE(LlamaBiHybridRetrieverForNCE):