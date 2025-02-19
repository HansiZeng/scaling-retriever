from copy import deepcopy

import torch 
import ujson 
import numpy as np 


def tokenize_add_cls_token_id_and_padding(tokenizer, texts, max_length):
    assert tokenizer.padding_side == "left", tokenizer.padding_side
    tokenized_texts = tokenizer(texts,
                                truncation=True, 
                                padding=False,
                                max_length=max_length-1,
                                return_attention_mask=False,
                                add_special_tokens=True)
    tokenized_texts["input_ids"] = [ids + [tokenizer.cls_token_id] for ids in tokenized_texts["input_ids"]]
    tokenized_texts = tokenizer.pad(tokenized_texts,
                                    padding=True,
                                    pad_to_multiple_of=8,
                                    return_attention_mask=True,
                                    return_tensors="pt")
    return tokenized_texts


class T5SpladeCollatorForNCE:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
    
    def __call__(self, batch):
        queries, pos_texts, batch_neg_texts = [list(xs) for xs in zip(*batch)]
        
        tokenized_queries = self.tokenizer(queries, 
                                            max_length=self.query_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        texts = pos_texts + [neg_txt for neg_txts in batch_neg_texts for neg_txt in neg_txts]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.doc_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        labels = torch.arange(0, len(queries))
        
        tokenized_queries["decoder_input_ids"] = deepcopy(tokenized_queries["input_ids"])
        tokenized_contexts["decoder_input_ids"] = deepcopy(tokenized_contexts["input_ids"])

        return {
            "tokenized_queries": tokenized_queries,
            "tokenized_contexts": tokenized_contexts,
            "target_labels": labels, # we don't name it "labels", seems it might arise bugs using Huggingface Trainer
        }


class LlamaSpladeCollatorForNCE:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
    
    def __call__(self, batch):
        queries, pos_texts, batch_neg_texts = [list(xs) for xs in zip(*batch)]
        
        tokenized_queries = self.tokenizer(queries, 
                                            max_length=self.query_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        texts = pos_texts + [neg_txt for neg_txts in batch_neg_texts for neg_txt in neg_txts]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.doc_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        labels = torch.arange(0, len(queries))
        # print("size of query, texts, labels: ", len(queries), len(texts), len(labels))

        return {
            "tokenized_queries": tokenized_queries,
            "tokenized_contexts": tokenized_contexts,
            "target_labels": labels, # we don't name it "labels", seems it might arise bugs using Huggingface Trainer
        }
LlamaDenseCollatorForNCE = LlamaSpladeCollatorForNCE


class LlamaSpladeCollatorForKLDiv:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        
    def __call__(self, batch):
        queries, pos_texts, batch_neg_texts, pos_score, neg_scores = [list(xs) for xs in zip(*batch)]
        
        tokenized_queries = self.tokenizer(queries, 
                                            max_length=self.query_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        texts = []
        for pos_text, neg_texts in zip(pos_texts, batch_neg_texts):
            texts.extend([pos_text] + neg_texts)
        tokenized_contexts = self.tokenizer(texts, 
                                            max_length=self.doc_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        batch_size, num_neg = len(queries), len(batch_neg_texts[0])
        teacher_scores = [] 
        for p_score, n_scores in zip(pos_score, neg_scores):
            teacher_scores.append([p_score] + n_scores)
        teacher_scores = torch.FloatTensor(teacher_scores)
        assert teacher_scores.shape == (batch_size, num_neg + 1)  
        
        return {
            "tokenized_queries": tokenized_queries,
            "tokenized_contexts": tokenized_contexts,
            "teacher_scores": teacher_scores,
        }
LlamaDenseCollatorForKLDiv = LlamaSpladeCollatorForKLDiv


class LlamaSpladeCollatorForNCE_KLDiv:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        
    def __call__(self, batch):
        queries, pos_texts, batch_neg_texts, pos_score, neg_scores = [list(xs) for xs in zip(*batch)]
        
        tokenized_queries = self.tokenizer(queries, 
                                            max_length=self.query_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        
        
        texts = pos_texts + [neg_txt for neg_txts in batch_neg_texts for neg_txt in neg_txts]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.doc_max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        labels = torch.arange(0, len(queries))
        
        # We also provide the teacher scores as the teacher_scores
        # teacher_scores's shape is [batch_size, num_neg_samples + 1]
        batch_size, num_neg = len(queries), len(batch_neg_texts[0])
        teacher_scores = [] 
        for p_score, n_scores in zip(pos_score, neg_scores):
            teacher_scores.append([p_score] + n_scores)
        teacher_scores = torch.FloatTensor(teacher_scores)
        assert teacher_scores.shape == (batch_size, num_neg + 1)  
        
        # When applying encode() in the model, query's shape is [batch_size, D]
        # and context's shape is [batch_size * (num_neg_samples + 1), D]. 
        # The result `logits` will be [batch_size, batch_size * (num_neg_samples + 1)]
        # Hence we should find the correpsonding indexes for the teacher_scores 
        teacher_idxes = [[i] + list(range(batch_size + i*num_neg, batch_size + (i+1)*num_neg)) for i in range(batch_size)]
        teacher_idxes = torch.LongTensor(teacher_idxes)
        assert teacher_idxes.shape == (batch_size, num_neg + 1)
        
        return {
            "tokenized_queries": tokenized_queries,
            "tokenized_contexts": tokenized_contexts,
            "target_labels": labels, # we don't name it "labels", seems it might arise bugs using Huggingface Trainer
            "teacher_scores": teacher_scores,
            "teacher_idxes": teacher_idxes
        }
LlamaDenseCollatorForNCE_KLDiv = LlamaSpladeCollatorForNCE_KLDiv


class T5SpladeCollectionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        ids, texts = [list(xs) for xs in zip(*batch)]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        tokenized_contexts["decoder_input_ids"] = deepcopy(tokenized_contexts["input_ids"])
        return {
            **{k: v for k, v in tokenized_contexts.items()},
            "ids": ids
        }


class LlamaSpladeCollectionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        ids, texts = [list(xs) for xs in zip(*batch)]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        return {
            **{k: v for k, v in tokenized_contexts.items()},
            "ids": ids
        }
        

LlamaDenseCollectionCollator = LlamaSpladeCollectionCollator
LlamaHybridCollectionCollator = LlamaSpladeCollectionCollator
        

class LlamaSpladeCollatorForMarginMSE: 
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.query_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt",
                                        pad_to_multiple_of=8)
        
        pos_tokenized_doc = self.tokenizer(list(pos_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.doc_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt",
                                        pad_to_multiple_of=8)
        
        neg_tokenized_doc = self.tokenizer(list(neg_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.doc_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt",
                                        pad_to_multiple_of=8)
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }


LlamaDenseCollatorForMarginMSE = LlamaSpladeCollatorForMarginMSE


class T5SpladeCollatorForMarginMSE:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.query_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        pos_tokenized_doc = self.tokenizer(list(pos_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.doc_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        neg_tokenized_doc = self.tokenizer(list(neg_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.doc_max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        tokenized_query["decoder_input_ids"] = deepcopy(tokenized_query["input_ids"])
        pos_tokenized_doc["decoder_input_ids"] = deepcopy(pos_tokenized_doc["input_ids"])
        neg_tokenized_doc["decoder_input_ids"] = deepcopy(neg_tokenized_doc["input_ids"])
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }
        
        
class HybridRetrieverRerankCollator:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
    
    def __call__(self, batch):
        qids, docids, queries, docs = [list(xs) for xs in zip(*batch)]
        
        tokenized_queries = self.tokenizer(queries,
                                           max_length=self.query_max_length,
                                           truncation=True, padding="longest", return_tensors="pt")
        tokenized_docs = self.tokenizer(docs,
                                        max_length=self.doc_max_length,
                                        truncation=True, padding="longest", return_tensors="pt")
        
        return {
            "qids": qids,
            "docids": docids,
            "tokenized_queries": tokenized_queries,
            "tokenized_docs": tokenized_docs
        }
        
        
class RerankerInferenceCollator:
    def __init__(self, tokenizer, max_length, pad_to_multiple_of=16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, batch):
        # pids and docids mean same thing
        qids, docids, text_pairs = [list(xs) for xs in zip(*batch)]
        
        collated_pairs = self.tokenizer(
            text_pairs,
            padding=False, 
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        
        collated_pairs = self.tokenizer.pad(
            collated_pairs,
            padding=True, 
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',)
        
        return {
            "qids": qids,
            "docids": docids,
            "tokenized_texts": collated_pairs
        }
        

class BertRerankerInferenceCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        qids, docids, queries, docs = [list(xs) for xs in zip(*batch)]
        collated_pairs = self.tokenizer(
            queries,
            docs,
            padding=True, 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        return {
            "qids": qids,
            "docids": docids,
            "tokenized_texts": collated_pairs
        }