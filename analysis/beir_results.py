import ujson 
import os 
import argparse 
from collections import defaultdict
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", default=None, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    datasets = [
    "arguana",
    "fiqa",
    "nfcorpus",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "hotpotqa",
    "nq"]
    
    miss_dss = []
    metric_to_result = defaultdict(list)
    for ds in datasets:
        try:
            with open(os.path.join(args.base_dir, ds, "perf.json")) as fin:
                perf = ujson.load(fin)
                for metric, value in perf.items():
                    metric_to_result[metric].append(value)
        except:
            miss_dss.append(ds)
            
    if len(miss_dss) > 0:
        print("missing datasets: ", miss_dss) 
    else:
        print("avg performance for BEIR: ")
        metric_to_result = {k: np.mean(v) for k, v in metric_to_result.items()}
        print(metric_to_result)
        with open(os.path.join(args.base_dir, "average_perf.json"), "w") as fout:
            ujson.dump(metric_to_result, fout, indent=4)