import faiss 
import os 
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    T5Config
)
from transformers import AutoConfig
import torch
import ujson
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from tqdm import tqdm
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

from scaling_retriever.modeling.llm_encoder import LlamaBiDense
from scaling_retriever.dataset.dataset import CollectionDataset, MSMARCOQueryDataset, BeirDataset
from scaling_retriever.dataset.data_collator import LlamaDenseCollectionCollator
from scaling_retriever.indexer import store_embs, DenseFlatIndexer
from scaling_retriever.utils.utils import is_first_worker, obtain_doc_vec_dir_files, supports_bfloat16
import constants
from scaling_retriever.utils.metrics import load_and_evaluate, evaluate_beir

def ddp_setup(args):
    init_process_group(backend="nccl")
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = torch.distributed.get_world_size()
    

@dataclass
class DenseRetrievalArguments:
    model_name_or_path: str = field(default=None)
    ckpt_path: str = field(default=None)
    model_type: str = field(default=None)
    corpus_path: str = field(default=None)
    doc_embed_dir: str = field(default=None)
    index_dir: str = field(default=None)
    out_dir: str = field(default=None)
    query_path: str = field(default=None)
    eval_run_path: str = field(default=None)
    eval_qrel_path: str = field(default=None)
    eval_metric: str = field(default=None)
    
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
    top_k: int = field(default=1000)
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)
    
    task_name: str = field(default="")

    def __post_init__(self):
        # assert self.task_name in ["write_doc_embeds", "retrieval", "indexing", "evaluate_msmarco"]
        if self.eval_metric: 
            self.eval_metric = eval(self.eval_metric)
            print("evaluation info: ", self.eval_qrel_path, self.eval_run_path, self.eval_metric)     
            
        if self.is_beir:
            self.doc_max_length == 512 and self.query_max_length == 512      
        

def load_from_adapter(model_name_or_path):
    assert os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = ujson.load(f)
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    print("load lora model from ", model_name_or_path) 
    model = LlamaBiDense.load(base_model_name_or_path, 
                            lora_name_or_path=model_name_or_path)
    return model
        

class DenseRetriever:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device
    
    def generate_query_vecs(self, dataloader):
        model = self.model
        all_query_reps = []
        all_qids = []
        
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = {k:v.to(self.device) for k,v in batch.items() if k != "ids"}
            with torch.no_grad():
                query_reps = model.query_encode(**inputs).cpu().numpy()
            all_query_reps.append(query_reps)
            all_qids.extend(batch["ids"])
                
        return np.concatenate(all_query_reps, axis=0), all_qids
    
class LocalFaissDenseRetriever(DenseRetriever):
    def __init__(self, model, device, index):
        super().__init__(model, device)
        self.index = index 
        
    def index_encoded_data(self, doc_vec_files, doc_id_files):
        doc_reps = []
        doc_ids = []
        for doc_file, id_file in zip(doc_vec_files, doc_id_files):
            doc_reps.append(np.load(doc_file))
            doc_ids.append(np.load(id_file))
            
        doc_reps = np.concatenate(doc_reps, axis=0)
        doc_ids = np.concatenate(doc_ids).tolist()
        
        assert len(doc_reps) == len(doc_ids), (len(doc_reps), len(doc_ids))
        
        print("size of doc reps to index: ", doc_reps.shape)
        self.index.index_data(doc_reps, doc_ids)
        print("finished indexing")
        
    def get_top_docs(self, dataloader, top_docs):
        query_reps, qids = self.generate_query_vecs(dataloader)
        top_doc_ids, top_scores = self.index.search_knn(query_reps, top_docs)
        
        assert len(qids) == len(query_reps), (len(qids), len(query_reps))
        
        return (qids, top_doc_ids, top_scores)


def evaluate_msmarco(args):
    res = {}
    for metric in args.eval_metric:
        metric_val = load_and_evaluate(args.eval_qrel_path, args.eval_run_path, metric)
        res[metric] = metric_val
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "perf.json"), "w") as fout:
        ujson.dump(res, fout, indent=4)
        
        
def main():
    parser = HfArgumentParser((DenseRetrievalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.task_name not in  ["evaluate_msmarco", "evaluate_beir"]:
        ddp_setup(args)
        assert args.local_rank != -1
        device = args.local_rank
        print("world_size = {}, local_rank = {}".format(args.world_size, args.local_rank))
     
    if args.task_name == "write_doc_embeds":
        if is_first_worker():
            os.makedirs(args.doc_embed_dir, exist_ok=True)
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.is_beir and args.beir_dataset is not None:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
            data_path = util.download_and_unzip(url, args.beir_dataset_dir)
            
            corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
            collection_dataset = BeirDataset(corpus, information_type="document")
        else:
            collection_dataset = CollectionDataset(corpus_path=args.corpus_path,
                                                data_source=constants.corpus_datasource[args.corpus_path])
        data_collator = LlamaDenseCollectionCollator(tokenizer=tokenizer, 
                                                    max_length=args.doc_max_length)
        dataloader = DataLoader(dataset=collection_dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False, num_workers=1,
                                sampler=DistributedSampler(collection_dataset, shuffle=False),
                                collate_fn=data_collator)
        model = LlamaBiDense.load_from_lora(args.model_name_or_path)
        model.to(device)
        model.eval()
        
        tokenizer.padding_side = "left"
        assert tokenizer.pad_token == tokenizer.eos_token
        
        store_embs(model=model, collection_loader=dataloader, local_rank=args.local_rank,
                   index_dir=args.doc_embed_dir, device=device)    
    elif args.task_name == "retrieval":
        assert args.world_size == 1 and args.local_rank == 0, (args.world_size, args.local_rank)
        os.makedirs(args.out_dir, exist_ok=True)
        index = DenseFlatIndexer()
        if args.index_dir is not None:
            index.deserialize(args.index_dir)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            print("hidden_size: ", config.hidden_size)
            index.init_index(config.hidden_size)
        
        model = LlamaBiDense.load_from_lora(args.model_name_or_path)
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left"
        #assert tokenizer.cls_token == "<|eot_id|>", tokenizer.cls_token
        assert tokenizer.pad_token == tokenizer.eos_token
        
        if args.is_beir and args.beir_dataset is not None:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
            data_path = util.download_and_unzip(url, args.beir_dataset_dir)
            
            _, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
            query_dataset = BeirDataset(queries, information_type="query")
        else:
            query_dataset = MSMARCOQueryDataset(args.query_path)
        query_collator = LlamaDenseCollectionCollator(tokenizer=tokenizer,
                                                      max_length=args.query_max_length)
        query_dataloader = DataLoader(query_dataset, batch_size=args.eval_batch_size, shuffle=False, 
                                      num_workers=4, collate_fn=query_collator)
        
        retriever = LocalFaissDenseRetriever(model, index=index, device=device)
        doc_vector_files, doc_id_files = obtain_doc_vec_dir_files(args.doc_embed_dir)
        
        retriever.index_encoded_data(doc_vector_files, doc_id_files)
        qids, top_doc_ids, top_doc_scores = retriever.get_top_docs(query_dataloader, top_docs=args.top_k)
        
        qid_to_rankdata = {}
        for qid, doc_ids, doc_scores in zip(qids, top_doc_ids, top_doc_scores):
            qid = str(qid)
            for docid, score in zip(doc_ids, doc_scores):
                docid = str(docid)
                score = float(score)
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = {docid: score}
                else:
                    qid_to_rankdata[qid][docid] = score
                    
        with open(os.path.join(args.out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    elif args.task_name == "evaluate_msmarco":
        evaluate_msmarco(args)
    elif args.task_name == "evaluate_beir":
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.beir_dataset}.zip"
        data_path = util.download_and_unzip(url, args.beir_dataset_dir)
        
        _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        evaluate_beir(args, qrels)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()

    