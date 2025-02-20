import torch 
import numpy as np 
np.set_printoptions(suppress=True, precision=3)

from transformers import AutoTokenizer 
from scaling_retriever.modeling.llm_encoder import LlamaBiSparse, LlamaBiDense

# LlamaBiSparse.load_from_lora(lora_name_or_path=model_name_or_path)

# load sparse model and tokenizer
model = LlamaBiDense.load_from_lora("hzeng/Lion-DS-1B-llama3-marco-mntp") 
tokenizer = AutoTokenizer.from_pretrained( "hzeng/Lion-DS-1B-llama3-marco-mntp")


queries = ["What is the capital of France?", "Who wrote '1984'?"]
passages = [
    "Paris is the capital of France.",
    "George Orwell wrote '1984'."
]
tokenized_queries = tokenizer(queries,
                                max_length=192,
                                truncation=True, padding="longest", return_tensors="pt")
tokenized_passages = tokenizer(passages,
                                max_length=192,
                                truncation=True, padding="longest", return_tensors="pt")

quey_embeds = model.query_encode(**tokenized_queries)
doc_embeds = model.doc_encode(**tokenized_passages)

scores = torch.matmul(quey_embeds, doc_embeds.T)
print(scores.tolist())