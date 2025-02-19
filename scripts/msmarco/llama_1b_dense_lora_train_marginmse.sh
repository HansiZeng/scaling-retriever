#!/bin/bash

# train 
corpus_path=/work/yyy/ir-research/data/msmarco-full/full_collection/raw.tsv
train_path=/work/yyy/ir-research/llm_as_retriever_data/data/msmarco_qrel_added_query_teacher_scores.jsonl

model_name_or_path="/gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/mntp/llama3-1b-msmarco/bimodel" 

# lr, batch_size, epochs, effective_batch_size
list_of_tuples=(
    1e-4 256 56 1024
)
NGPU=4

for (( i=0; i<${#list_of_tuples[@]}; i+=3 )); do
    lr=${list_of_tuples[i]}
    batch_size=${list_of_tuples[i+1]}
    num_train_epochs=${list_of_tuples[i+2]}
    effective_batch_size=${list_of_tuples[i+3]}

    gradient_accumulation_steps=$(($effective_batch_size / $NGPU / $batch_size))
    
    echo gradient_accumulation_steps: $gradient_accumulation_steps
    echo epcohs: $num_train_epochs

    run_name=llama3-1b-marco-mntp-dense-mgmse-lora-${lr}_bs_${batch_size}_ep_${num_train_epochs}_l2norm
    output_dir=/gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/$run_name

    torchrun --nproc_per_node=$NGPU --master_port 4212 -m train_dense \
            --num_train_epochs=$num_train_epochs \
            --run_name=$run_name \
            --learning_rate=$lr \
            --model_name_or_path=$model_name_or_path \
            --output_dir=$output_dir \
            --wandb_project_name=llm_as_retriever \
            --bf16 \
            --query_max_length=64 \
            --doc_max_length=128 \
            --per_device_train_batch_size=$batch_size \
            --gradient_accumulation_steps=$gradient_accumulation_steps \
            --corpus_path=$corpus_path \
            --train_path=$train_path \
            --loss_type=margin_mse \
            --logging_steps 50 \
            --warmup_ratio 0.04 \
            --save_strategy epoch \
            --save_total_limit=1 \
            --fsdp "full_shard auto_wrap" \
            --train_config /work/yyy/ir-research/llm_as_retriever/train_configs/llama_config.json \
            --gradient_checkpointing \
            --fsdp_config /work/yyy/ir-research/llm_as_retriever/train_configs/fsdp_llama_config.json \
            --lora \
            --model_type llama
done