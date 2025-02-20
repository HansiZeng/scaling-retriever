#/bin/bash

task_name=index_and_retrieval
data_root_dir=/gypsum/work1/xxx/yyy/llm_as_retriever
corpus_path=/work/yyy/ir-research/PAG/data/msmarco-full/full_collection/raw.tsv
if [ $task_name = index_and_retrieval ]; then 
    # index 
    list_model_name_paths=(
        $data_root_dir/checkpoints/llama3-3b-marco-mntp-dense-mgmse-lora-1e-4_bs_128_ep_24_l2norm
    )
    for model_name_or_path in "${list_model_name_paths[@]}"; do
        echo "task_name: $task_name"
        torchrun --nproc_per_node=2 --master_port 3408 -m eval_dense \
            --model_name_or_path $model_name_or_path \
            --doc_embed_dir $model_name_or_path/doc_embeds \
            --task_name write_doc_embeds \
            --eval_batch_size 128 \
            --corpus_path $corpus_path

        # retrieval
        query_paths=(
            /work/yyy/ir-research/PAG/data/msmarco-full/dev_queries/raw.tsv
            /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv
            /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv
        )

        for query_path in "${query_paths[@]}"; do
            if [[ $query_path == *all-train.csv ]]; then 
                set_name=train
                ds_name=all
            elif [[ $query_path == *queries_2019/raw.tsv ]]; then 
                set_name=test
                ds_name=trec_dl_19
            elif [[ $query_path == *queries_2020/raw.tsv ]]; then 
                set_name=test
                ds_name=trec_dl_20
            elif [[ $query_path == *dev_queries/raw.tsv ]]; then 
                set_name=dev
                ds_name=msmarco
            else 
                echo "Error: Unknown set_name: $set_name"
                exit 1
            fi
            out_dir=$model_name_or_path/all_retrieval/${ds_name}/${set_name}

            torchrun --nproc_per_node=1 --master_port 44450 -m eval_dense \
                --model_name_or_path $model_name_or_path \
                --doc_embed_dir $model_name_or_path/doc_embeds \
                --out_dir $out_dir \
                --query_path $query_path \
                --task_name retrieval \
                --top_k 1000
        
            if [[ $ds_name == msmarco && $set_name == dev ]]; then
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/dev_qrel.json
                eval_metric='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_dense \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metric \
                    --out_dir $out_dir \
                    --task evaluate_msmarco
            elif [[ $ds_name == trec_dl_19 && $set_name == test ]]; then 
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/run.json
                python -m eval_dense \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task evaluate_msmarco

                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_dense \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir ${out_dir}_binary \
                    --task evaluate_msmarco
            elif [[ $ds_name == trec_dl_20 && $set_name == test ]]; then 
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/run.json
                python -m eval_dense \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task evaluate_msmarco

                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_dense \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir ${out_dir}_binary \
                    --task evaluate_msmarco
            else 
                echo "Error: Unknown dataset: $ds_name"
                exit 1
            fi 

            if [[ $set_name == train ]]; then 
                python preprocess/create_de_self_train_data.py \
                    --q_ppid_npids_path "/work/yyy/ir-research/GR-for-RAG-data/data/dpr-all/all-ret/query_pospid_negpids.train.jsonl" \
                    --retrieval_path $out_dir/rankings.${set_name}.jsonl \
                    --example_path $out_dir/de.train.jsonl
            fi
        done
        python analysis/beir_results.py \
                --base_dir $model_name_or_path/beir/all_retrieval
    done
fi
