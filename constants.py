
query_path_datasource = {
    "/work/hzeng_umass_edu/ir-research/PAG/data/msmarco/train_queries/labeled_queries/raw.tsv": "msmarco",
    "./data/msmarco-full/dev_queries/raw.tsv": "msmarco",
    "./data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv": "msmarco",
    "./data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv": "msmarco",
}

corpus_datasource = {
    "./data/msmarco-full/full_collection/raw.tsv": "msmarco",

}

supported_models = ["t5", "llama", "bert"]
