# *Scaling Sparse and Dense Retrieval in Decoder-Only LLMs*


# Environment

## Getting Start
### Preparing the Models
We provide two retrieval paradigms: sparse retrieval and dense retrieval. For sparse models:
```python
from transformers import AutoTokenizer 
from scaling_retriever.modeling.llm_encoder import LlamaBiSparse

model = LlamaBiSparse.load_from_lora("hzeng/Lion-SP-1B-llama3-marco-mntp") 
tokenizer = AutoTokenizer.from_pretrained( "hzeng/Lion-SP-1B-llama3-marco-mntp")
```
For dense models:
```python
from transformers import AutoTokenizer 
from scaling_retriever.modeling.llm_encoder import LlamaBiDense

model = LlamaBiDense.load_from_lora("hzeng/Lion-DS-1B-llama3-marco-mntp") 
tokenizer = AutoTokenizer.from_pretrained( "hzeng/Lion-DS-1B-llama3-marco-mntp")
```

### Inference (Toy example)
```python
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

# sparse retrieval scores:
# [
#    [14.835160255432129, 0.026406031101942062], 
#    [0.005473464727401733, 13.909822463989258]
# ]

# dense retrieval scores:
# [
#    [0.2877607047557831, 0.13211995363235474],    
#    [0.1040663793683052, 0.29219019412994385]
# ]
```


## Evaluation
# MSMARCO

# BEIR


## Training