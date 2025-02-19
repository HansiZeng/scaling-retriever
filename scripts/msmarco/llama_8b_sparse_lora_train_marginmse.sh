#!/bin/bash

# It taks 260 hours on 4 A100 for around 12.5 epochs

extract_task_weights() {
    local task_weights=$1

    # Remove the brackets and extract the individual values
    weights=$(echo $task_weights | tr -d '[]' | awk -F, '{print $2, $3}')

    # Assign the extracted values to the new variables and remove leading '.'
    local query_reg=$(echo $weights | awk '{print $1}' | sed 's/^\.//')
    local doc_reg=$(echo $weights | awk '{print $2}' | sed 's/^\.//')

    # Return the values as a string
    echo "$query_reg $doc_reg"
}

# train 
corpus_path=/work/yyy/ir-research/data/msmarco-full/full_collection/raw.tsv
train_path=/work/yyy/ir-research/llm_as_retriever_data/data/msmarco_qrel_added_query_teacher_scores.jsonl

model_name_or_path="/work/yyy/ir-research/llm_as_retriever_data/checkpoints/mntp/llama3-8b-msmarco"

echo $teacher_score_path

# lr, task_weights, batch_size effective_batch_size epochs
list_of_tuples=(
    1e-4 '[1.,.05,.04]' 64 512 8
)
NGPU=4
# 4 A100, ep=8, bs=64, ef_bs=512 takes 40hours - 48hours (roughly 2 days)

for (( i=0; i<${#list_of_tuples[@]}; i+=3 )); do
    lr=${list_of_tuples[i]}
    task_weights=${list_of_tuples[i+1]}
    batch_size=${list_of_tuples[i+2]}
    effective_batch_size=${list_of_tuples[i+3]}
    epochs=${list_of_tuples[i+4]}

    gradient_accumulation_steps=$(($effective_batch_size / $NGPU / $batch_size))

    if [[ $effective_batch_size == 512 ]]; then 
        max_steps=$((1050 * $epochs))
    elif [[ $effective_batch_size == 1024 ]]; then
        max_steps=$((525 * $epochs))
    else 
        echo "effective_batch_size is not supported"
        exit 1
    fi
    save_steps=$((max_steps / 5))

    echo gradient_accumulation_steps: $gradient_accumulation_steps
    echo max_steps: $max_steps
    echo epcohs: $epochs

    read query_reg doc_reg <<< $(extract_task_weights "$task_weights")
    run_name=llama3-8b-marco-mntp-sparse-mgmse-lora-${lr}_qreg_${query_reg}_dreg_${doc_reg}_bs_${batch_size}_ep_${epochs}
    output_dir=/gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/$run_name

    torchrun --nproc_per_node=$NGPU --master_port 4426 -m train_splade \
            --max_steps=$max_steps \
            --run_name=$run_name \
            --learning_rate=$lr \
            --model_name_or_path=$model_name_or_path \
            --output_dir=$output_dir \
            --task_names='["rank","query_reg","doc_reg"]' \
            --task_weights=$task_weights \
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
            --save_steps $save_steps \
            --save_total_limit=1 \
            --fsdp "full_shard auto_wrap" \
            --train_config /work/yyy/ir-research/llm_as_retriever/train_configs/llama_config.json \
            --gradient_checkpointing \
            --fsdp_config /work/yyy/ir-research/llm_as_retriever/train_configs/fsdp_llama_config.json \
            --lora \
            --model_type llama
done