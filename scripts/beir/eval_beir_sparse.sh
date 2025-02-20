#/bin/bash

task_name=index_and_retrieval
data_root_dir=/gypsum/work1/xxx/yyy/llm_as_retriever
beir_dataset_dir=/gypsum/work1/xxx/yyy/llm_as_retriever/beir_datasets
if [ $task_name = index_and_retrieval ]; then 
    list_of_tuples=(
       $data_root_dir/checkpoints/llama3-1b-marco-mntp-sparse-kldiv-lora-1e-4_qreg_05_dreg_04_bs_28_epochs_7_nnegs_16
       $data_root_dir/checkpoints/llama3-8b-marco-mntp-sparse-kldiv-lora-1e-4_qreg_05_dreg_04_bs_8_epochs_1_nnegs_16
    )
    for model_name_or_path in "${list_of_tuples[@]}"; do
        for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq; do
            echo model_name_or_path $model_name_or_path
            echo dataset $dataset
            # index 
            torchrun --nproc_per_node=4 --master_port 4432 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir $model_name_or_path/beir/${dataset}/index \
                --task_name indexing \
                --eval_batch_size 32 \
                --is_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --query_max_length 512 \
                --doc_max_length 512

            python -m utils.inverted_index \
                --model_name_or_path $model_name_or_path \
                --index_dir $model_name_or_path/beir/${dataset}

            rm -rf $model_name_or_path/beir/${dataset}/index_0 
            rm -rf $model_name_or_path/beir/${dataset}/index_1
            rm -rf $model_name_or_path/beir/${dataset}/index_2
            rm -rf $model_name_or_path/beir/${dataset}/index_3

            # retrieval 
            out_dir=$model_name_or_path/beir/all_retrieval/${dataset}

            torchrun --nproc_per_node=1 --master_port 44450 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir $model_name_or_path/beir/${dataset}/index \
                --out_dir $out_dir \
                --task_name retrieval \
                --top_k 100 \
                --is_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --query_max_length 512 \
                --doc_max_length 512 \
                --eval_batch_size 64

            python -m eval_sparse \
                --task evaluate_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --out_dir $out_dir
        done
        python analysis/beir_results.py \
            --base_dir $model_name_or_path/beir/all_retrieval 
    done 
fi