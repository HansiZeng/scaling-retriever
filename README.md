# LLMs for Sparse and Dense Retrieval: A Comparison of the Scaling Behavior of Different Retrieval Paradigms
This is the repo for a SIGIR 2025 short paper submission.

## Pre-traning
run ```bash scripts/run_llama_mntp.sh```

## Fine-tuning
All fine-tuning files are located in `scripts/msmarco` with the naming convention: ```llama[Size]b_[RetParadigm]_lora_train_[FTObjective].sh```
  - Size: 1, 3, 8.
  - RetParadigm stands for "retrieval paradigm", you can select: sparse and dense.
  - FTObjective stands for "fine-tuning objective", you can select: kd, cl, cl-kd.

For example, the 8B sparse model trained with CL+KD loss is named `llama_8b_sparse_lora_train_kd.sh`
