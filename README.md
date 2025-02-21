# *Scaling Sparse and Dense Retrieval in Decoder-Only LLMs*


# Environment Setup
To use scaling_retriever, first install the requirement packages:
```bash
pip install -r requirements.txt
conda install -c pytorch faiss-cpu=1.8.0
```


## Getting Start
### Preparing the Models
We provide two retrieval paradigms: sparse retrieval and dense retrieval. 
For sparse models:
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
Before running the evaluation scripts, **download the required data** from the following link:  
🔗 [MSMARCO Evaluation and Training Data](https://drive.google.com/drive/folders/1KVbSr7yO6Uig6YEJeSBHgrRMLcEhGOc9?usp=sharing)  

Once downloaded, **place the files in the current directory** to ensure proper access during evaluation.  

### MSMARCO

#### Evaluate the 1B Sparse model
The evaluation benchmarks for MSMARCO include MSMARCO Dev, TREC DL 2019, and TREC DL 2020.
To evaluate the 1B sparse model (hzeng/Lion-SP-1B-llama3-marco-mntp), run:

```bash scripts/eval_sparse.sh```
#### Evaluate the 8B Sparse model
To evaluate the 8B sparse model, modify line 7 in `scripts/eval_sparse.sh`:
Change the model name to: `hzeng/Lion-SP-8B-llama3-marco-mntp`
Then re-run the script:

```bash scripts/eval_sparse.sh```
#####  ⚠ **Warning: CPU Usage for Evaluation**  
For efficient evaluation, **please ensure that you use more than 32 CPUs**, as using fewer cores may significantly slow down retrieval.  

Our implementation utilizes **multi-threading for retrieval in an inverted index**, and an insufficient number of CPUs may lead to unexpected performance issues.  

Expected runtime: On **MS MARCO Dev**, retrieval typically completes in **~15 minutes** under optimal CPU conditions.  

#### Evaluate the 1B Dense Model
To evaluate the 1B dense model (hzeng/Lion-DS-1B-llama3-marco-mntp), run:

```bash scripts/eval_dense.sh```
#### Evaluate the 8B Dense model
To evaluate the 8B sparse model, modify line 7 in `scripts/eval_dense.sh`:
Change the model name to: `hzeng/Lion-DS-8B-llama3-marco-mntp`
Then re-run the script:

```bash scripts/eval_dense.sh```
### BEIR

#### Evauate the Sparse Model

```bash scripts/beir/eval_beir_sparse.sh```

The default setting is to evaluate `hzeng/Lion-SP-1B-llama3-marco-mntp`, change `hzeng/Lion-SP-8B-llama3-marco-mntp` in line7 for the 8B model.
#### Evaluate the Dense Model

```bash scripts/beir/eval_beir_dense.sh```


The default setting is to evaluate `hzeng/Lion-DS-1B-llama3-marco-mntp`, change `hzeng/Lion-DS-8B-llama3-marco-mntp` in line7 for the 8B model.

## Training
TODO