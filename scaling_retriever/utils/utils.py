import os 
import itertools 
import random

import ujson 
import torch.distributed
import torch


random.seed(1234)

def has_answer(text, answers):
    for answer in answers:
        text = text.strip().lower().replace(' ', '')
        answer = answer.strip().lower().replace(' ', '')
        if text.find(answer) != -1:
            return True
    return False

def is_first_worker():
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def obtain_doc_vec_dir_files(doc_embed_dir):
    with open(os.path.join(doc_embed_dir, "plan.json")) as fin:
        plan = ujson.load(fin)
    nranks = plan["nranks"]
    num_chunks = plan["num_chunks"]
    
    doc_vec_files = []
    doc_id_files = []
    for i in range(nranks):
        for j in range(num_chunks):
            vec_file = os.path.join(doc_embed_dir, f"embs_{i}_{j}.npy")
            doc_id_file = os.path.join(doc_embed_dir, f"ids_{i}_{j}.npy")
            assert os.path.exists(vec_file) and os.path.exists(doc_id_file)
            
            doc_vec_files.append(vec_file)
            doc_id_files.append(doc_id_file)
    
    return doc_vec_files, doc_id_files

def sum_to_main(x):
    if not torch.distributed.is_initialized():
        return x 
    
    assert torch.distributed.get_world_size() > 1
    summed_x = torch.distributed.reduce(x, op=torch.distributed.ReduceOp.SUM)

    return summed_x

def distributed_weighted_average(avg_value, count, device):
    if not torch.distributed.is_initialized():
        return avg_value
    
    assert torch.distributed.get_world_size() > 1
    total_value = torch.tensor([avg_value * count], device=device)
    total_count = torch.tensor([count], device=device)
    
    summed_total_value = sum_to_main(total_value)
    summed_total_count = sum_to_main(total_count)
    weighted_avg = summed_total_value / summed_total_count
    
    return weighted_avg

# Function to check if current GPU supports bfloat16 (bf16)
def supports_bfloat16():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device)
        # Check if the GPU supports bf16
        return device_props.major >= 8  # GPUs with compute capability >= 8.0 support bf16 (like A100)
    return False


def batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def get_data_source(args):
    if "msmarco" in args.corpus_path and "msmarco" in args.train_path:
        return "msmarco" 
    elif "wiki" in args.corpus_path and "wiki" in args.train_path:
        return "wiki"
    else:
        raise NotImplementedError

