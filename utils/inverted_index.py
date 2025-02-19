import array
import json
import os
import pickle
from collections import defaultdict
from typing import List
import argparse

import h5py
import numpy as np
from tqdm.auto import tqdm
import ujson


class IndexDictOfArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                if isinstance(doc_ids, list):
                    self.n = len(doc_ids)
                else:
                    min_val = 1000000000 
                    max_val = -100000000
                    for val in doc_ids:
                        min_val = min(min_val, val)
                        max_val = max(max_val, val)
                    print("min_val: ", min_val, "max_val: ", max_val)
                    assert min_val == 0, min_val 
                    self.n = max_val + 1
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """add a batch of documents to the index
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        # for debug 
        print("filename: ", self.filename)
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
        

def merge_indexes(model_name_or_path, filename="array_index.h5py", index_name="index", index_dir=None):
    with open(os.path.join(model_name_or_path, "config.json")) as fin: 
        config = ujson.load(fin)
    dim_voc = config["vocab_size"]
    print("dim_voc: ", dim_voc)
    
    if index_dir is not None:
        index_dirs = [os.path.join(index_dir, d) for d in os.listdir(index_dir)
                      if d.startswith(index_name)]
    else:
        index_dirs = [os.path.join(model_name_or_path, d) for d in os.listdir(model_name_or_path)
                    if d.startswith(index_name)]
    assert len(index_dirs) in [1, 2, 4], index_dirs 
    
    if len(index_dirs) == 1:
        print("only one index, no need to merge")
        return
    
    index_doc_id = dict() 
    index_doc_value = dict()
    doc_ids = dict()
    index_dist = {}
    index_stats = {"L0_d": 0}
    for idx_dir in index_dirs:
        file = h5py.File(os.path.join(idx_dir, filename), "r")
        assert file["dim"][()] <= dim_voc, (file["dim"][()], dim_voc)
        for key in tqdm(range(dim_voc)):
            try:
                new_index_doc_id = np.array(file["index_doc_id_{}".format(key)], dtype=np.int32)
                new_index_doc_value = np.array(file["index_doc_value_{}".format(key)], dtype=np.float32)
            except:
                new_index_doc_id = np.array([], dtype=np.int32)
                new_index_doc_value = np.array([], dtype=np.float32)
            if key not in index_doc_id:
                index_doc_id[key] = new_index_doc_id
                index_doc_value[key] = new_index_doc_value
            else:
                index_doc_id[key] = np.append(index_doc_id[key], new_index_doc_id)
                index_doc_value[key] = np.append(index_doc_value[key], new_index_doc_value)
        
        with open(os.path.join(idx_dir, "doc_ids.pkl"), "rb") as f:
            doc_ids.update(pickle.load(f))
        with open(os.path.join(idx_dir, "index_dist.json"), "r") as f:
            index_dist.update(json.load(f))
        with open(os.path.join(idx_dir, "index_stats.json"), "r") as f:
            index_stats["L0_d"] += json.load(f)["L0_d"] / len(index_dirs)

    if index_dir is not None:
        out_index_dir = os.path.join(index_dir, index_name)
    else:
        out_index_dir = os.path.join(model_name_or_path, index_name)
    os.makedirs(out_index_dir, exist_ok=True)
    with h5py.File(os.path.join(out_index_dir, filename), "w") as f:
        f.create_dataset("dim", data=int(dim_voc))
        for key in tqdm(index_doc_id.keys()):
            f.create_dataset("index_doc_id_{}".format(key), data=index_doc_id[key])
            f.create_dataset("index_doc_value_{}".format(key), data=index_doc_value[key])
    with open(os.path.join(out_index_dir, "doc_ids.pkl"), "wb") as f:
        pickle.dump(doc_ids, f)
    with open(os.path.join(out_index_dir, "index_dist.json"), "w") as f:
        json.dump(index_dist, f)
    with open(os.path.join(out_index_dir, "index_stats.json"), "w") as f:
        json.dump(index_stats, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--index_name", default="index", type=str)
    parser.add_argument("--index_dir", default=None, type=str)
    args = parser.parse_args()
    
    merge_indexes(args.model_name_or_path, index_name=args.index_name, index_dir=args.index_dir)