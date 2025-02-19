import ujson 
import gzip 
import os 
import random

from torch.utils.data import Dataset
import pandas as pd
import torch

from beir.datasets.data_loader import GenericDataLoader

def read_wiki_corpus(corpus_path):
        pid_to_doc = {}
        with open(corpus_path) as fin:
            for i, line in enumerate(fin):
                if i != 0:
                    pid, text, title = line.strip().split("\t")
                    pid_to_doc[pid] = (title, text)
        return pid_to_doc
    
def read_msmarco_corpus(corpus_path):
    pid_to_doc = {}
    with open(corpus_path) as fin:
        for line in fin:
            pid, text = line.strip().split("\t")
            pid_to_doc[pid] = (None, text)
    return pid_to_doc 

def read_msmarco_query(query_path):
    qid_to_query = {}
    with open(query_path) as fin:
        for line in fin:
            qid, query = line.strip().split("\t")
            qid_to_query[qid] = query
    return qid_to_query


def get_doc_text(title, text):
        if title is None:
            return text
        else:
            return f"title: {title} | context: {text}"

class DualEncoderDatasetForNCE(Dataset):
    def __init__(self, corpus_path, train_path, data_source, n_negs=1):
        if data_source == "wiki":
            self.pid_to_doc = read_wiki_corpus(corpus_path)
        elif data_source == "msmarco":
            self.pid_to_doc = read_msmarco_corpus(corpus_path)
        else:
            raise ValueError("data_source must be either wiki or msmarco")
        print("size of corpus = {}".format(len(self.pid_to_doc)))
        
        self.examples = []
        with open(train_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                
                query, pos_pid, neg_pids = example["question"], example["pos_pid"], example["neg_pids"]
                self.examples.append((query, pos_pid, neg_pids))
        self.n_negs = n_negs
        self.data_source = data_source
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, pos_pid, neg_pids = self.examples[idx]
        
        if self.data_source == "wiki":
            # for wiki data, there exits some cases that the negative samples are not enough
            # in this case, we will sample with replacement
            sample_neg_pids = random.choices(neg_pids, k=self.n_negs) if len(neg_pids) < self.n_negs else \
                random.sample(neg_pids, k=self.n_negs)
        else:
            sample_neg_pids = random.sample(neg_pids, k=self.n_negs)
        
        pos_text = get_doc_text(*self.pid_to_doc[pos_pid])
        neg_texts = [] 
        for neg_pid in sample_neg_pids:
            neg_texts.append(get_doc_text(*self.pid_to_doc[neg_pid]))
        
        return (
            query,
            pos_text,
            neg_texts
        )
        

class DualEncoderDatasetForMarginMSE(Dataset):
    def __init__(self, corpus_path, train_path, data_source):
        if data_source == "wiki":
            self.pid_to_doc = read_wiki_corpus(corpus_path)
        elif data_source == "msmarco":
            self.pid_to_doc = read_msmarco_corpus(corpus_path)
        else:
            raise ValueError("data_source must be either wiki or msmarco")
        print("size of corpus = {}".format(len(self.pid_to_doc)))
        
        self.examples = []
        with open(train_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(example)
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, docids, scores = self.examples[idx]["query"], self.examples[idx]["docids"], self.examples[idx]["scores"]

        pos_docid = docids[0]
        pos_score = scores[0]

        neg_idx = random.sample(range(1, len(docids)), k=1)[0]
        neg_docid = docids[neg_idx]
        neg_score = scores[neg_idx]
        
        pos_doc = get_doc_text(*self.pid_to_doc[pos_docid]) 
        neg_doc = get_doc_text(*self.pid_to_doc[neg_docid])

        return query, pos_doc, neg_doc, pos_score, neg_score
    

class DualEncoderDatasetForKLDiv(Dataset):
    def __init__(self, corpus_path, train_path, data_source, n_negs=1):
        if data_source == "msmarco":
            self.pid_to_doc = read_msmarco_corpus(corpus_path)
        else:
            raise ValueError("data_source must be either wiki or msmarco")
        print("size of corpus = {}".format(len(self.pid_to_doc)))
        
        self.examples = []
        with open(train_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                
                query, pos_pid, neg_pids, pos_score, neg_scores \
                    = example["question"], example["pos_pid"], example["neg_pids"], \
                        example["pos_score"], example["neg_scores"]
                self.examples.append((query, pos_pid, neg_pids, pos_score, neg_scores))
        self.n_negs = n_negs
        self.data_source = data_source
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, pos_pid, neg_pids, pos_score, neg_scores = self.examples[idx]
        
        assert len(neg_pids) == len(neg_scores)
        sample_neg_idxes = random.sample(range(len(neg_pids)), k=self.n_negs)
        sample_neg_pids = [neg_pids[i] for i in sample_neg_idxes]
        sample_neg_scores = [neg_scores[i] for i in sample_neg_idxes]
        
        pos_text = get_doc_text(*self.pid_to_doc[pos_pid])
        neg_texts = [] 
        for neg_pid in sample_neg_pids:
            neg_texts.append(get_doc_text(*self.pid_to_doc[neg_pid]))
        
        return (
            query,
            pos_text,
            neg_texts,
            pos_score,
            sample_neg_scores,
        )
        
        
class CollectionDataset(Dataset):
    def __init__(self, corpus_path, data_source=None):
        if data_source == "msmarco":
            self.pid_to_doc = read_msmarco_corpus(corpus_path)
        elif data_source == "wiki":
            self.pid_to_doc = read_wiki_corpus(corpus_path)
        else:
            raise NotImplemented
        self.pids = list(self.pid_to_doc.keys())
        
    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        text = get_doc_text(*self.pid_to_doc[pid])
        
        return pid, text
    
    
class WikiQueryDataset(Dataset):
    def __init__(self, query_path):
        query_data = pd.read_csv(query_path, sep="\t", names=["question", "answers"])
        self.queries = list(query_data.question)
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # wiki dont' have qies 
        # We use the query itself as qid 
        # the qid is in the first field
        return self.queries[idx], self.queries[idx]


class MSMARCOQueryDataset(Dataset):
    def __init__(self, query_path):
        # query and corpus sets have same format in msmarco
        self.qid_to_query = read_msmarco_query(query_path)
        self.qids = list(self.qid_to_query.keys())
        
    def __len__(self):
        return len(self.qid_to_query)
    
    def __getitem__(self, idx):
        qid = self.qids[idx]
        query = self.qid_to_query[qid]
        
        return qid, query
    
    
class HybridRetrieverRerankDataset(Dataset):
    def __init__(self,
                 qid_pid_pairs,
                 query_path,
                 corpus_path,
                 data_source=None):
        self.qid_pid_pairs = qid_pid_pairs
        self.query_path = query_path
        self.corpus_path = corpus_path
        
        if data_source == "msmarco":
            self.pid_to_doc = read_msmarco_corpus(corpus_path)
            self.qid_to_query = read_msmarco_query(query_path)
        elif data_source == "wiki":
            self.pid_to_doc = read_wiki_corpus(corpus_path)
    
    def __len__(self):
        return len(self.qid_pid_pairs)
    
    def __getitem__(self, idx):
        qid, pid = self.qid_pid_pairs[idx]
        query = self.qid_to_query[qid]
        doc = get_doc_text(*self.pid_to_doc[pid])
        
        return qid, pid, query, doc
    
    
class RerankerInferenceDataset(Dataset):
    def __init__(self, 
                 qid_pid_pairs, 
                 query_path, 
                 corpus_path,
                 query_prefix=None,
                 doc_prefix=None):
        self.qid_pid_pairs = qid_pid_pairs
        self.qid_to_query = read_msmarco_query(query_path)
        self.pid_to_doc = read_msmarco_corpus(corpus_path)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        
    def _pair_format(self, query, doc, query_prefix, doc_prefix):
        if query_prefix is not None and doc_prefix is not None:
            return f"{query_prefix} {query} {doc_prefix} {doc}"
        
    def __len__(self):
        return len(self.qid_pid_pairs)
    
    def __getitem__(self, idx):
        qid, pid = self.qid_pid_pairs[idx]
        query = self.qid_to_query[qid]
        doc = get_doc_text(*self.pid_to_doc[pid])
        text_pair = self._pair_format(query, doc, self.query_prefix, self.doc_prefix)
        
        return qid, pid, text_pair
    

class BertRerankerInferenceDataset(Dataset):
    def __init__(self, 
                 qid_pid_pairs, 
                 query_path, 
                 corpus_path):
        self.qid_pid_pairs = qid_pid_pairs
        self.qid_to_query = read_msmarco_query(query_path)
        self.pid_to_doc = read_msmarco_corpus(corpus_path)
    
    def __len__(self):
        return len(self.qid_pid_pairs)
    
    def __getitem__(self, idx):
        qid, pid = self.qid_pid_pairs[idx]
        query = self.qid_to_query[qid]
        doc = get_doc_text(*self.pid_to_doc[pid])
        
        return qid, pid, query, doc
    

class BeirDataset(Dataset):
    """
    dataset to iterate over a BEIR collection
    we preload everything in memory at init
    """

    def __init__(self, value_dictionary, information_type="document"):
        assert information_type in ["document", "query"]
        self.value_dictionary = value_dictionary
        self.information_type = information_type
        if self.information_type == "document":
            self.value_dictionary = dict()
            for key, value in value_dictionary.items():
                self.value_dictionary[key] = value["title"] + " " + value["text"]
        self.idx_to_key = {idx: key for idx, key in enumerate(self.value_dictionary)}

    def __len__(self):
        return len(self.value_dictionary)

    def __getitem__(self, idx):
        true_idx = self.idx_to_key[idx]
        return true_idx, self.value_dictionary[true_idx]


class BeirRerankDataset(Dataset):
    def __init__(self, data_path, qid_docid_pairs):
        corpus, queries, _  = GenericDataLoader(data_folder=data_path).load(split="test")
        
        self.key_to_doc = {}
        for key, value in corpus.items():
            self.key_to_doc[key] = value["title"] + " " + value["text"]
        self.key_to_query = queries 
        
        self.qid_docid_pairs = qid_docid_pairs
                
    def __len__(self):
        return len(self.qid_docid_pairs)
    
    def __getitem__(self, idx):
        qid, docid = self.qid_docid_pairs[idx]
        query = self.key_to_query[qid]
        doc = self.key_to_doc[docid]
        
        return qid, docid, query, doc
        