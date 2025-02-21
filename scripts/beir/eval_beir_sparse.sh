#/bin/bash

task_name=index_and_retrieval
beir_dataset_dir=./data/beir_datasets
if [ $task_name = index_and_retrieval ]; then 
    list_of_tuples=(
       hzeng/Lion-SP-1B-llama3-marco-mntp
    )
    for model_name_or_path in "${list_of_tuples[@]}"; do
        for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq; do
            echo model_name_or_path $model_name_or_path
            echo dataset $dataset
            # index 
            torchrun --nproc_per_node=2 --master_port 4432 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir ./output/$model_name_or_path/beir/${dataset}/index \
                --task_name indexing \
                --eval_batch_size 32 \
                --is_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --query_max_length 512 \
                --doc_max_length 512

            python -m utils.inverted_index \
                --model_name_or_path $model_name_or_path \
                --index_dir ./output/$model_name_or_path/beir/${dataset}

            rm -rf ./output/$model_name_or_path/beir/${dataset}/index_0 
            rm -rf ./output/$model_name_or_path/beir/${dataset}/index_1
            rm -rf ./output/$model_name_or_path/beir/${dataset}/index_2
            rm -rf ./output/$model_name_or_path/beir/${dataset}/index_3

            # retrieval 
            out_dir=./output/$model_name_or_path/beir/all_retrieval/${dataset}

            torchrun --nproc_per_node=1 --master_port 44450 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir ./output/$model_name_or_path/beir/${dataset}/index \
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
            --base_dir ./output/$model_name_or_path/beir/all_retrieval 
    done 
fi