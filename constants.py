
query_path_datasource = {
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco/train_queries/labeled_queries/raw.tsv": "msmarco",
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco-full/dev_queries/raw.tsv": "msmarco",
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv": "msmarco",
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv": "msmarco",
    "/work/hzeng_umass_edu/ir-research/GR-for-RAG-data/data/dpr-all/qas/nq-dev.csv": "nq"
}

corpus_datasource = {
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco-full/full_collection/raw.tsv": "msmarco",
    "/work/hzeng_umass_edu/ir-research/GR-for-RAG-data/data/NQ/psgs_w100.tsv": "nq"
}

supported_models = ["t5", "llama", "bert"]


peft_model_base_model_map = {
    "castorini/rankllama-v1-7b-lora-passage": "meta-llama/Llama-2-7b-hf"
}