#/bin/bash

task_name=index_and_retrieval
data_root_dir=/gypsum/work1/xxx/yyy/llm_as_retriever
beir_dataset_dir=/gypsum/work1/xxx/yyy/llm_as_retriever/beir_datasets
if [ $task_name = index_and_retrieval ]; then 
    list_of_tuples=(
        $data_root_dir/checkpoints/llama3-8b-marco-mntp-dense-kldiv-lora-1e-4_bs_8_ep_1_nnegs_16_l2norm
    )
    for model_name_or_path in "${list_of_tuples[@]}"; do
        for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq; do
            echo model_name_or_path $model_name_or_path
            echo dataset $dataset
            
            # index 
            torchrun --nproc_per_node=4 --master_port 4436 -m eval_dense \
                --model_name_or_path $model_name_or_path \
                --doc_embed_dir $model_name_or_path/beir/${dataset}/doc_embeds \
                --task_name write_doc_embeds \
                --eval_batch_size 64 \
                --is_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --query_max_length 512 \
                --doc_max_length 512

            out_dir=$model_name_or_path/beir/all_retrieval/${dataset}

            torchrun --nproc_per_node=1 --master_port 44450 -m eval_dense \
                --model_name_or_path $model_name_or_path \
                --doc_embed_dir $model_name_or_path/beir/${dataset}/doc_embeds \
                --out_dir $out_dir \
                --task_name retrieval \
                --top_k 100 \
                --is_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --query_max_length 512 \
                --doc_max_length 512 \
                --eval_batch_size 64

            python -m eval_dense \
                --task evaluate_beir \
                --beir_dataset $dataset \
                --beir_dataset_dir $beir_dataset_dir \
                --out_dir $out_dir
        done
        python analysis/beir_results.py \
                --base_dir $model_name_or_path/beir/all_retrieval
    done
fi