#/bin/bash

task_name=index_and_retrieval
data_root_dir=/gypsum/work1/xxx/yyy/llm_as_retriever
if [ $task_name = indexing ]; then 
    echo "task_name: $task_name"
    torchrun --nproc_per_node=1 --master_port 4432 -m eval_splade \
        --model_name_or_path $model_name_or_path \
        --index_dir $model_name_or_path/index \
        --task_name indexing \
        --eval_batch_size 64 \
        --corpus_path  /work/yyy/ir-research/PAG/data/msmarco-full/full_collection/raw.tsv

    python -m utils.inverted_index \
        --model_name_or_path $model_name_or_path
elif [ $task_name = retrieval ]; then 
     echo "task_name: $task_name"
     query_paths=(
        /work/yyy/ir-research/PAG/data/msmarco-full/dev_queries/raw.tsv
        /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv
        /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv
    )

    list_of_tuples=(
        $model_name_or_path
    )
    for model_name_or_path in "${list_of_tuples[@]}"; do
        for query_path in "${query_paths[@]}"; do
            if [[ $query_path == *nq-dev.csv ]]; then 
                set_name=dev
                ds_name=nq
            elif [[ $query_path == *nq-test.csv ]]; then 
                set_name=test
                ds_name=nq
            elif [[ $query_path == *trivia-dev.csv ]]; then 
                set_name=dev
                ds_name=trivia
            elif [[ $query_path == *webq-test.csv ]]; then 
                set_name=test
                ds_name=webq
            elif [[ $query_path == *curatedtrec-test.csv ]]; then 
                set_name=test
                ds_name=curatedtrec
            elif [[ $query_path == *all-train.csv ]]; then 
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

            torchrun --nproc_per_node=1 --master_port 44450 -m eval_splade \
                --model_name_or_path $model_name_or_path \
                --index_dir $model_name_or_path/index \
                --out_dir $out_dir \
                --query_path $query_path \
                --task_name retrieval \
                --top_k 1000

            case "$query_path" in
                "*nq-dev.csv"|"*nq-test.csv"|"*trivia-dev.csv"|"*webq-test.csv"|"*curatedtrec-test.csv")
                    # for evaluate DPR provided evaluation set
                    python preprocess/trans_json_to_jsonl_ranking.py \
                    --in_dir $out_dir 

                    python -m eval_rankings \
                        --retrieval_path $out_dir/rankings.${set_name}.jsonl \
                        --eval_path $query_path \
                        --out_dir $out_dir
                    ;;
                "*queries_2019/raw.tsv"|"*queries_2020/raw.tsv"|"*dev_queries/raw.tsv")
                    # for evaluate MSMARCO provided evaluation set
                    
            esac
            
            python preprocess/trans_json_to_jsonl_ranking.py \
                --in_dir $out_dir 

            if [[ $ds_name == msmarco && $set_name == dev ]]; then
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/dev_qrel.json
                eval_metric='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_splade \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metric \
                    --out_dir $out_dir \
                    --task evaluate_msmarco
            elif [[ $ds_name == trec_dl_19 && $set_name == test ]]; then 
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/run.json
                python -m eval_splade \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task evaluate_msmarco

                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_splade \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir ${out_dir}_binary \
                    --task evaluate_msmarco
            elif [[ $ds_name == trec_dl_20 && $set_name == test ]]; then 
                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/run.json
                python -m eval_splade \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task evaluate_msmarco

                eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/run.json
                python -m eval_splade \
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
    done
elif [ $task_name = index_and_retrieval ]; then
    list_model_name_paths=(
        $data_root_dir/checkpoints/llama3-1b-marco-mntp-sparse-nce-kldiv-lora-1e-4_qreg_05_dreg_04_bs_28_epochs_7_nnegs_16_sp8b_cnt_teacher
        $data_root_dir/checkpoints/llama3-8b-marco-mntp-sparse-nce-kldiv-lora-1e-4_qreg_05_dreg_04_bs_8_epochs_1_nnegs_16_lorar_32
    )
    for model_name_or_path in "${list_model_name_paths[@]}"; do
        # index
        echo "task_name: $task_name"
        torchrun --nproc_per_node=2 --master_port 4432 -m eval_splade \
            --model_name_or_path $model_name_or_path \
            --index_dir $model_name_or_path/index \
            --task_name indexing \
            --eval_batch_size 64 \
            --corpus_path  /work/yyy/ir-research/PAG/data/msmarco-full/full_collection/raw.tsv

        python -m utils.inverted_index \
            --model_name_or_path $model_name_or_path

        # retrieval 
        echo "task_name: $task_name"
        query_paths=(
            /work/yyy/ir-research/PAG/data/msmarco-full/dev_queries/raw.tsv
            /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv
            /work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv
        )
        list_of_tuples=(
            $model_name_or_path
        )
        for model_name_or_path in "${list_of_tuples[@]}"; do
            for query_path in "${query_paths[@]}"; do
                if [[ $query_path == *nq-dev.csv ]]; then 
                    set_name=dev
                    ds_name=nq
                elif [[ $query_path == *nq-test.csv ]]; then 
                    set_name=test
                    ds_name=nq
                elif [[ $query_path == *trivia-dev.csv ]]; then 
                    set_name=dev
                    ds_name=trivia
                elif [[ $query_path == *webq-test.csv ]]; then 
                    set_name=test
                    ds_name=webq
                elif [[ $query_path == *curatedtrec-test.csv ]]; then 
                    set_name=test
                    ds_name=curatedtrec
                elif [[ $query_path == *all-train.csv ]]; then 
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

                torchrun --nproc_per_node=1 --master_port 44450 -m eval_splade \
                    --model_name_or_path $model_name_or_path \
                    --index_dir $model_name_or_path/index \
                    --out_dir $out_dir \
                    --query_path $query_path \
                    --task_name retrieval \
                    --top_k 1000

                case "$query_path" in
                    "*nq-dev.csv"|"*nq-test.csv"|"*trivia-dev.csv"|"*webq-test.csv"|"*curatedtrec-test.csv")
                        # for evaluate DPR provided evaluation set
                        python preprocess/trans_json_to_jsonl_ranking.py \
                        --in_dir $out_dir 

                        python -m eval_rankings \
                            --retrieval_path $out_dir/rankings.${set_name}.jsonl \
                            --eval_path $query_path \
                            --out_dir $out_dir
                        ;;
                    "*queries_2019/raw.tsv"|"*queries_2020/raw.tsv"|"*dev_queries/raw.tsv")
                        # for evaluate MSMARCO provided evaluation set      
                esac
                
                python preprocess/trans_json_to_jsonl_ranking.py \
                    --in_dir $out_dir 

                if [[ $ds_name == msmarco && $set_name == dev ]]; then
                    eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/dev_qrel.json
                    eval_metric='["mrr_10","recall"]'
                    eval_run_path=$out_dir/run.json
                    python -m eval_splade \
                        --eval_run_path $eval_run_path \
                        --eval_qrel_path $eval_qrel_path \
                        --eval_metric $eval_metric \
                        --out_dir $out_dir \
                        --task evaluate_msmarco
                elif [[ $ds_name == trec_dl_19 && $set_name == test ]]; then 
                    eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel.json
                    eval_metrics='["ndcg_cut"]'
                    eval_run_path=$out_dir/run.json
                    python -m eval_splade \
                        --eval_run_path $eval_run_path \
                        --eval_qrel_path $eval_qrel_path \
                        --eval_metric $eval_metrics \
                        --out_dir $out_dir \
                        --task evaluate_msmarco

                    eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2019/qrel_binary.json
                    eval_metrics='["mrr_10","recall"]'
                    eval_run_path=$out_dir/run.json
                    python -m eval_splade \
                        --eval_run_path $eval_run_path \
                        --eval_qrel_path $eval_qrel_path \
                        --eval_metric $eval_metrics \
                        --out_dir ${out_dir}_binary \
                        --task evaluate_msmarco
                elif [[ $ds_name == trec_dl_20 && $set_name == test ]]; then 
                    eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel.json
                    eval_metrics='["ndcg_cut"]'
                    eval_run_path=$out_dir/run.json
                    python -m eval_splade \
                        --eval_run_path $eval_run_path \
                        --eval_qrel_path $eval_qrel_path \
                        --eval_metric $eval_metrics \
                        --out_dir $out_dir \
                        --task evaluate_msmarco

                    eval_qrel_path=/work/yyy/ir-research/PAG/data/msmarco-full/TREC_DL_2020/qrel_binary.json
                    eval_metrics='["mrr_10","recall"]'
                    eval_run_path=$out_dir/run.json
                    python -m eval_splade \
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
        done
    done
fi
