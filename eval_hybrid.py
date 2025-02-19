# import faiss
import os
import torch.distributed
import ujson 
from dataclasses import field, dataclass

import transformers
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd 
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import supports_bfloat16

from dataset.dataset import CollectionDataset, NQQueryDataset, MSMARCOQueryDataset
from dataset.data_collator import  LlamaHybridCollectionCollator
from modeling.llm_encoder import LlamaBiHybridRetrieverForNCE as LlamaBiHybridRetriever
from utils.utils import is_first_worker, obtain_doc_vec_dir_files, supports_bfloat16
from indexer import HybridIndexer, HybridRetriever
import constants
from utils.metrics import load_and_evaluate


def ddp_setup(args):
    init_process_group(backend="nccl")
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = torch.distributed.get_world_size()
    

@dataclass
class HybridRetrievalArguments:
    model_name_or_path: str = field(default=None)
    ckpt_path: str = field(default=None)
    model_type: str = field(default=None)
    corpus_path: str = field(default=None)
    doc_embed_dir: str = field(default=None)
    dense_index_dir: str = field(default=None)
    sparse_index_dir: str = field(default=None)
    out_dir: str = field(default=None)
    query_path: str = field(default=None)
    eval_run_path: str = field(default=None)
    eval_qrel_path: str = field(default=None)
    eval_metric: str = field(default=None)
    
    eval_batch_size: int = field(default=128)
    doc_max_length: int = field(default=192)
    query_max_length: int = field(default=64) 
    hidden_dim: int = field(default=768)
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)
    top_k: int = field(default=1000)
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)
    
    task_name: str = field(default="")

    def __post_init__(self):
        assert self.task_name in ["write_doc_embeds", "retrieval", "indexing", "evaluate_msmarco"]
        if self.eval_metric: 
            self.eval_metric = eval(self.eval_metric)
            print("evaluation info: ", self.eval_qrel_path, self.eval_run_path, self.eval_metric)           
        

def load_from_adapter(model_name_or_path, model_cls):
    assert os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = ujson.load(f)
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    print("load lora model from ", model_name_or_path) 
    model = model_cls.load(base_model_name_or_path, 
                            lora_name_or_path=model_name_or_path)
    return model


def hybrid_index(args, model_type):
    ddp_setup(args)
    args.world_size = torch.distributed.get_world_size()
    print("local_rank = {}, world_size = {}".format(args.local_rank, args.world_size))
    device = args.local_rank
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    
    if is_first_worker():
        os.makedirs(args.sparse_index_dir, exist_ok=True)
        os.makedirs(args.dense_index_dir, exist_ok=True)
    
    if model_type == "llama":
        d_collection = CollectionDataset(corpus_path=args.corpus_path, 
                                        data_source=constants.corpus_datasource[args.corpus_path])
        d_collator = LlamaHybridCollectionCollator(tokenizer=tokenizer, max_length=args.doc_max_length)
        dataloader = DataLoader(dataset=d_collection,
                                    batch_size=args.eval_batch_size,
                                    shuffle=False, num_workers=2,
                                    sampler=DistributedSampler(d_collection, shuffle=False),
                                    collate_fn=d_collator)
        model = load_from_adapter(args.model_name_or_path, model_cls=LlamaBiHybridRetriever)
    
    model.to(device)
    model.eval()
    
    if torch.distributed.get_world_size() > 1:
        sparse_index_dir = args.sparse_index_dir[:-1] if args.sparse_index_dir.endswith("/") else args.sparse_index_dir
        sparse_index_dir = f"{sparse_index_dir}_{torch.distributed.get_rank()}"
    else:
        sparse_index_dir = args.sparse_index_dir
    print(sparse_index_dir, args.local_rank, model.vocab_size)
    
    indexer = HybridIndexer(model=model,
                            sparse_index_dir=sparse_index_dir,
                            dense_index_dir=args.dense_index_dir,
                            tokenizer=tokenizer,
                            device=device,
                            compute_stats=True,
                            dim_voc=model.vocab_size)
    indexer.index(dataloader)
    

def hybrid_retrieval(args, model_type):
    ddp_setup(args)
    assert args.world_size == 1 and args.local_rank == 0, (args.world_size, args.local_rank)
    device = args.local_rank
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    #assert tokenizer.cls_token == "<|eot_id|>", tokenizer.cls_token
    assert tokenizer.pad_token == tokenizer.eos_token
    
    query_dataset = MSMARCOQueryDataset(args.query_path)
    
    if model_type == "llama":
        query_collator = LlamaHybridCollectionCollator(tokenizer=tokenizer,
                                                        max_length=args.query_max_length)
        query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size, shuffle=False, 
                                        num_workers=4, collate_fn=query_collator)
        model = load_from_adapter(args.model_name_or_path, model_cls=LlamaBiHybridRetriever)
    model.to(device)
    model.eval()
        
    retriever = HybridRetriever(model=model,
                                sparse_index_dir=args.sparse_index_dir,
                                dense_index_dir=args.dense_index_dir,
                                out_dir=args.out_dir,
                                dim_voc=model.vocab_size,
                                device=device)
    retriever.retrieve(query_dataloader, topk=args.top_k, threshold=0.0)


def evaluate_msmarco(args):
    res = {}
    for metric in args.eval_metric:
        metric_val = load_and_evaluate(args.eval_qrel_path, args.eval_run_path, metric)
        res[metric] = metric_val
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "perf.json"), "w") as fout:
        ujson.dump(res, fout, indent=4)
    

if __name__ == "__main__":
    parser = HfArgumentParser((HybridRetrievalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    # Identify model_type
    if args.task_name != "evaluate_msmarco":
        with open(os.path.join(args.model_name_or_path, "config.json"), "r") as f:
            model_config = ujson.load(f)
        model_type = model_config["model_type"] 
        assert model_type in constants.supported_models, model_type
    
    if args.task_name == "indexing":
        hybrid_index(args, model_type)
    elif args.task_name == "retrieval":
        hybrid_retrieval(args, model_type)
    elif args.task_name == "evaluate_msmarco":
        evaluate_msmarco(args)
    else:
        raise NotImplementedError
            
    


