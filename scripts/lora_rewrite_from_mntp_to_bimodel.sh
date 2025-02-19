#!/bin/bash 


python preprocess/lora_rewrite_from_mntp_to_bimodel.py \
    --input_dir /gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/mntp/llama3-3b-msmarco \
    --output_dir /gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/mntp/llama3-3b-msmarco/bimodel 