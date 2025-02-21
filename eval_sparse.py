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
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import hf_hub_download

from scaling_retriever.dataset.dataset import CollectionDataset, WikiQueryDataset, MSMARCOQueryDataset, BeirDataset
from scaling_retriever.dataset.data_collator import T5SparseCollectionCollator, LlamaSparseCollectionCollator
from scaling_retriever.modeling.llm_encoder import T5Sparse, LlamaBiSparse
from scaling_retriever.modeling.losses.regulariaztion import L0, FLOPS 
from scaling_retriever.utils.utils import supports_bfloat16
from scaling_retriever.indexer import SparseIndexer, SparseRetrieval
import constants
from scaling_retriever.utils.metrics import load_and_evaluate, evaluate_beir


def ddp_setup(args):
    init_process_group(backend="nccl")
    args.local_rank = int(os.environ["LOCAL_RANK"])

@dataclass
class SparseArguments:
    model_name_or_path: str = field(default=None)
    corpus_path: str = field(default="")
    index_dir: str = field(default=None)
    out_dir: str = field(default=None)
    query_path: str = field(default=None)
    retrieval_path: str = field(default=None)
    eval_path: str = field(default=None)
    data_source: str = field(default="msmarco")
    lora_name_or_path: str = field(default=None)
    
    # beir datasets
    is_beir: bool = field(default=False)
    beir_dataset: str = field(default=None)
    beir_dataset_dir: str = field(default=None)
    
    eval_batch_size: int = field(default=128)
    doc_max_length: int = field(default=192)
    query_max_length: int = field(default=64) 
    hidden_dim: int = field(default=768)
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)
    top_k: int = field(default=100)
    bow_topk: int = field(default=64)
    
    task_name: str = field(default="")
    eval_qrel_path: str = field(default="")
    eval_run_path: str = field(default="")
    eval_metric: str = field(default="")

    def __post_init__(self):
        # assert self.task_name in ["retrieval", "indexing", "evaluate_msmarco"]
        if self.eval_metric: 
            self.eval_metric = eval(self.eval_metric)
            print("evaluation info: ", self.eval_qrel_path, self.eval_run_path, self.eval_metric)    
        
        if self.is_beir:
            self.doc_max_length == 512 and self.query_max_length == 512         
        
        
def sparse_index(args, model_type):
    ddp_setup(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.is_beir and args.beir_dataset is not None:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
        data_path = util.download_and_unzip(url, args.beir_dataset_dir)
        
        corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
        d_collection = BeirDataset(corpus, information_type="document")
    else:
        d_collection = CollectionDataset(corpus_path=args.corpus_path, 
                                         data_source=constants.corpus_datasource[args.corpus_path])
    if model_type == "t5":
        model = T5Sparse.load(args.model_name_or_path)
        d_collator = T5SparseCollectionCollator(tokenizer=tokenizer, max_length=args.doc_max_length)
    elif model_type == "llama":
        print(args.model_name_or_path)
        model = LlamaBiSparse.load_from_lora(args.model_name_or_path)
        d_collator = LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=args.doc_max_length)
    d_loader = DataLoader(d_collection, batch_size=args.eval_batch_size, shuffle=False, 
                          collate_fn=d_collator, num_workers=2,
                          sampler=DistributedSampler(d_collection, shuffle=False))
    
    if torch.distributed.get_world_size() > 1:
        index_dir = args.index_dir[:-1] if args.index_dir.endswith("/") else args.index_dir
        index_dir = f"{index_dir}_{torch.distributed.get_rank()}"
    else:
        index_dir = args.index_dir
    print(index_dir, args.local_rank, model.vocab_size)
    indexer = SparseIndexer(model, index_dir=index_dir, compute_stats=True, dim_voc=model.vocab_size,
                            device=args.local_rank if args.local_rank != -1 else 0)
    indexer.index(d_loader)
    
    
def sparse_retrieval(args, model_type):
    ddp_setup(args)
    args.world_size = torch.distributed.get_world_size()
    print("local_rank = {}, world_size = {}".format(args.local_rank, args.world_size))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    assert args.world_size == 1, args.world_size
    
    if args.is_beir and args.beir_dataset is not None:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
        data_path = util.download_and_unzip(url, args.beir_dataset_dir)
        
        _, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
        q_collection = BeirDataset(queries, information_type="query")
    else:
        if constants.query_path_datasource[args.query_path] == "wiki":
            q_collection = WikiQueryDataset(args.query_path)
        elif constants.query_path_datasource[args.query_path] == "msmarco":
            q_collection = MSMARCOQueryDataset(args.query_path)
    
    if model_type == "t5":
        model = T5Sparse.load(args.model_name_or_path)
        q_collator = T5SparseCollectionCollator(tokenizer=tokenizer, max_length=args.query_max_length)
    elif model_type == "llama":
        model = LlamaBiSparse.load_from_lora(args.model_name_or_path)
        q_collator = LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=args.query_max_length)
    if args.world_size > 1:
        #assert args.world_size in [1,4], args.world_size
        q_loader = DataLoader(q_collection, batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=q_collator, num_workers=1,
                              sampler=DistributedSampler(q_collection, shuffle=False))
    else:
        q_loader = DataLoader(q_collection, batch_size=args.eval_batch_size, shuffle=False, 
                            collate_fn=q_collator, num_workers=4)
    
    config = {
        "index_dir": args.index_dir,
        "out_dir": args.out_dir
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    retriever = SparseRetrieval(config=config, model=model, compute_stats=True, 
                                dim_voc=model.vocab_size, device=args.local_rank if args.local_rank != -1 else 0)
    retriever.retrieve(q_loader, topk=args.top_k, threshold=0.0)
                       #name=None if args.local_rank == -1 else f"{args.local_rank}")
    

def evaluate_msmarco(args):
    res = {}
    for metric in args.eval_metric:
        metric_val = load_and_evaluate(args.eval_qrel_path, args.eval_run_path, metric)
        res[metric] = metric_val
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "perf.json"), "w") as fout:
        ujson.dump(res, fout, indent=4)
        

    
if __name__ == "__main__":
    parser = HfArgumentParser((SparseArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    # Identify model_type
    if args.task_name not in ["evaluate_msmarco", "evaluate_beir"]:
        if os.path.isdir(args.model_name_or_path): 
            with open(os.path.join(args.model_name_or_path, "config.json"), "r") as f:
                model_config = ujson.load(f)
        else:
            config_path = hf_hub_download(args.model_name_or_path, "config.json")
            with open(config_path, "r") as f:
                model_config = ujson.load(f)
        model_type = model_config["model_type"] 
        assert model_type in constants.supported_models, model_type
    
    if args.task_name == "indexing":
        sparse_index(args, model_type)
    elif args.task_name == "retrieval":
        sparse_retrieval(args, model_type)
    elif args.task_name == "evaluate_msmarco":
        evaluate_msmarco(args)
    elif args.task_name == "evaluate_beir":
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
        data_path = util.download_and_unzip(url, args.beir_dataset_dir)
        
        _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        evaluate_beir(args, qrels)
    else:
        raise NotImplementedError