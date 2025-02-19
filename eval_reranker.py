import os 
import ujson 
from dataclasses import field, dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from peft import PeftModel, PeftConfig
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from beir import util

from dataset.dataset import (
    RerankerInferenceDataset, 
    HybridRetrieverRerankDataset, 
    BertRerankerInferenceDataset,
    BeirRerankDataset
)
from dataset.data_collator import RerankerInferenceCollator, HybridRetrieverRerankCollator, BertRerankerInferenceCollator
from modeling.llm_encoder import  LlamaBiDense, LlamaBiSplade
import constants 


def ddp_setup(args):
    init_process_group(backend="nccl")
    args.local_rank = int(os.environ["LOCAL_RANK"])
    

def get_peft_crossencoder(peft_model_name, args):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, 
                                                                    num_labels=1,
                                                                    token=args.access_token)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

def load_adapter(model_name_or_path, model_cls):
    assert os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = ujson.load(f)
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    print("load lora model from ", model_name_or_path) 
    model = model_cls.load(base_model_name_or_path, 
                            lora_name_or_path=model_name_or_path)
    return model, base_model_name_or_path


@dataclass
class RerankArguments:
    model_name_or_path: str = field(default=None)
    peft_model_name: str = field(default=None)
    query_path: str = field(default=None)
    corpus_path: str = field(default=None)
    jsonl_path: str = field(default=None)
    run_path: str = field(default=None)
    output_dir: str = field(default=None)
    
    # beir datasets
    is_beir: bool = field(default=False)
    beir_dataset: str = field(default=None)
    beir_dataset_dir: str = field(default=None)
    
    query_max_length: int = field(default=64)
    doc_max_length: int = field(default=160)
    eval_batch_size: int = field(default=64)
    max_length: int = field(default=160)
    pad_to_multiple_of: int = field(default=16)
    rerank_type: str = field(default="hybrid_retriever")
    
    query_prefix: Optional[str] = field(default=None)
    doc_prefix: Optional[str] = field(default=None)
    access_token: Optional[str] = field(default=None)
    
    local_rank: int = field(default=-1)
    
    def __post_init__(self):
        assert not all([self.jsonl_path is not None, self.run_path is not None])
        assert not all([self.model_name_or_path is not None, self.peft_model_name is not None])
        assert self.rerank_type in ["hybrid_retriever", "dense_encoder", "splade", "cross_encoder"]
        
        
def main(args):
    ddp_setup(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    qid_docid_pairs = []
    if args.jsonl_path:
        # this is used for train_file: qrel_added_teacher_scores.json
        with open(args.jsonl_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                qid, docids = example["qid"], example["docids"]
                for docid in docids:
                    qid_docid_pairs.append((qid, docid))
    else:
        # this is for evaluation where we provide run.json
        with open(args.run_path) as fin:
            run_data = ujson.load(fin) 
        for qid, docid_score in run_data.items():
            for docid in docid_score:
                qid_docid_pairs.append((qid, docid))
    
    if args.rerank_type in ["hybrid_retriever", "dense_encoder", "splade"]:
        dataset = HybridRetrieverRerankDataset(qid_docid_pairs,
                                               query_path=args.query_path,
                                               corpus_path=args.corpus_path,
                                               data_source=constants.corpus_datasource[args.corpus_path])
    elif args.rerank_type == "cross_encoder":
        pass 
    else:
        raise NotImplementedError
    
    if args.peft_model_name is not None:
        if args.rerank_type in ["hybrid_retriever", "dense_encoder", "splade"]:
            if args.rerank_type == "hybrid_retriever":
                model_cls = LlamaBiHybridRetrieverForNCE
            elif args.rerank_type == "dense_encoder":
                model_cls = LlamaBiDense
            elif args.rerank_type == "splade":
                model_cls = LlamaBiSplade
                
            model, base_model_name_or_path = load_adapter(args.peft_model_name, model_cls)
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.peft_model_name)
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
            
            tokenizer.padding_side = "left"
            # assert tokenizer.pad_token == tokenizer.eos_token
            # tokenizer.add_special_tokens({"cls_token": "<|eot_id|>"})
            
            if args.rerank_type in ["hybrid_retriever", "dense_encoder", "splade"]:
                collator = HybridRetrieverRerankCollator(tokenizer=tokenizer,
                                                        query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length,)
        elif args.rerank_type == "cross_encoder":
            dataset = RerankerInferenceDataset(qid_docid_pairs, 
                                    query_path=args.query_path,
                                    corpus_path=args.corpus_path,
                                    query_prefix=args.query_prefix,
                                    doc_prefix=args.doc_prefix,)
            model = get_peft_crossencoder(args.peft_model_name, args)
            tokenizer_name_or_path = constants.peft_model_base_model_map[args.peft_model_name]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                    token=args.access_token)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = 0
            tokenizer.padding_side = 'right'
            collator = RerankerInferenceCollator(tokenizer=tokenizer,
                                                max_length=args.max_length,
                                                pad_to_multiple_of=args.pad_to_multiple_of)
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        if args.is_beir and args.beir_dataset is not None:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
            data_path = util.download_and_unzip(url, args.beir_dataset_dir)
            dataset = BeirRerankDataset(data_path, 
                                        qid_docid_pairs=qid_docid_pairs)
        else:
            assert args.model_name_or_path is not None
            dataset = BertRerankerInferenceDataset(qid_docid_pairs,
                                                query_path=args.query_path,
                                                corpus_path=args.corpus_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        collator = BertRerankerInferenceCollator(tokenizer=tokenizer,
                                                max_length=args.max_length)
    model.to(args.local_rank)
    model.eval()
    dataloader = DataLoader(dataset, 
                            batch_size=args.eval_batch_size, 
                            shuffle=False,
                            collate_fn=collator,
                            sampler=DistributedSampler(dataset, shuffle=False))                                 
    
    out_run = {}
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            if args.rerank_type in ["hybrid_retriever", "dense_encoder", "splade"]:
                batch["tokenized_queries"] = {k: v.to(args.local_rank) for k, v in batch["tokenized_queries"].items()}
                batch["tokenized_docs"] = {k: v.to(args.local_rank) for k, v in batch["tokenized_docs"].items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.rerank_forward(**batch)
            elif args.rerank_type == "cross_encoder":
                batch["tokenized_texts"] = {k: v.to(args.local_rank) for k, v in batch["tokenized_texts"].items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**batch["tokenized_texts"], return_dict=True).logits
            logits = logits.cpu().float().numpy()
            
            for i in range(len(logits)):
                qid = str(batch["qids"][i])
                docid = str(batch["docids"][i])
                if args.rerank_type in ["hybrid_retriever", "dense_encoder", "splade"]:
                    logit = float(logits[i])
                elif args.rerank_type == "cross_encoder":
                    logit = float(logits[i][0])
                if qid not in out_run:
                    out_run[qid] = {docid: logit}
                else:
                    out_run[qid][docid] = logit
                    
    
    if torch.distributed.get_world_size() == 1:
        assert args.local_rank <= 0 
        with open(os.path.join(args.output_dir, "run.json"), "w") as fout:
            ujson.dump(out_run, fout)
    else:
        with open(os.path.join(args.output_dir, f"run_{args.local_rank}.json"), "w") as fout:
            ujson.dump(out_run, fout)
        
if __name__ == "__main__":
    parser = HfArgumentParser((RerankArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    main(args)
                