import os 
import wandb
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Union
import logging

from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer,
                          BertForSequenceClassification, 
                          BertTokenizer,
                          HfArgumentParser,
                          TrainingArguments,
                          Trainer)
import transformers
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import EvalPrediction
from torch.utils.data.dataloader import DataLoader
import ujson
import torch
from copy import deepcopy
import numpy as np

TRAINING_ARGS_NAME = "training_args.bin"

from dataset.dataset import (
    DualEncoderDatasetForNCE, 
    DualEncoderDatasetForMarginMSE,
    DualEncoderDatasetForKLDiv
)
from dataset.data_collator import (
    T5SparseCollatorForNCE,
    T5SparseCollatorForMarginMSE,
    LlamaSparseCollatorForNCE,
    LlamaSparseCollatorForMarginMSE,
    LlamaSparseCollatorForNCE_KLDiv,
    LlamaSparseCollatorForKLDiv,
)
from modeling.llm_encoder import (
    T5Sparse, 
    LlamaBiSparse, 
    T5SparseForMarginMSE, 
    LlamaBiSparseForMarginMSE,
    LlamaBiSparseForNCE_KLDiv,
    LlamaBiSparseForKLDiv,
)
from tasks.sparse_trainer import LLM2RetrieverTrainingArgs, SparseTrainer
from modeling.losses.regulariaztion import RegWeightScheduler
from utils.utils import get_data_source

logger = logging.getLogger(__name__)

def save_training_args(model_args, args, output_dir):
    merged_args = {**asdict(model_args), **asdict(args)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as fout:
        ujson.dump(merged_args, fout, indent=4)
        
def get_model_type(model_name_or_path):
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    print("model_type: ", config.model_type) 
    return config.model_type

    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_from_adpater(args, model_cls):
    with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = ujson.load(f)
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    print("load lora model from ", args.model_name_or_path) 
    model = model_cls.load(base_model_name_or_path, 
                            lora_name_or_path=args.model_name_or_path,
                            merge_peft=False,
                            is_trainable=True)
    return model

if __name__ == "__main__":
    parser = HfArgumentParser((LLM2RetrieverTrainingArgs))
    args = parser.parse_args_into_dataclasses()[0]
    if args.local_rank <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as fout:
            ujson.dump(asdict(args), fout, indent=4)
            
    # Identify the model_type and either dense and sparse model 
    if args.model_type is None:
        model_type = get_model_type(args.model_name_or_path)
    else:
        model_type = args.model_type
    
    if args.loss_type == "nce":
        train_dataset= DualEncoderDatasetForNCE(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
            n_negs=args.n_negs
        )
    elif args.loss_type == "margin_mse":
        train_dataset = DualEncoderDatasetForMarginMSE(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
        )
    elif args.loss_type in ["nce_kldiv", "kldiv"]:
        train_dataset = DualEncoderDatasetForKLDiv(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
            n_negs=args.n_negs
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if model_type == "t5":
        if args.loss_type == "nce":
            train_collator =  T5SparseCollatorForNCE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            model = T5Sparse.build(args.model_name_or_path, args)
        elif args.loss_type == "margin_mse":
            train_collator = T5SparseCollatorForMarginMSE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            model = T5SparseForMarginMSE.build(args.model_name_or_path, args)
    elif model_type == "llama":
        if args.train_config is not None:
            with open(args.train_config, "r") as fin:
                config = ujson.load(fin)
        else:
            config = None
        if args.loss_type == "nce":
            train_collator = LlamaSparseCollatorForNCE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparse)
            else:
                model = LlamaBiSparse.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "margin_mse":
            train_collator = LlamaSparseCollatorForMarginMSE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForMarginMSE)
            else:
                model = LlamaBiSparseForMarginMSE.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "nce_kldiv":
            train_collator = LlamaSparseCollatorForNCE_KLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForNCE_KLDiv)
            else:
                model = LlamaBiSparseForNCE_KLDiv.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "kldiv":
            train_collator = LlamaSparseCollatorForKLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForKLDiv)
            else:
                model = LlamaBiSparseForKLDiv.build(args.model_name_or_path, args, config=config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # model might not be trainable after loading from lora 
        model.train()
        
    training_args = args
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    reg_to_reg_scheduler = {"doc_reg": RegWeightScheduler(lambda_=training_args.ln_to_weight["doc_reg"], 
                                                          T=training_args.max_steps // 3),
                            "query_reg": RegWeightScheduler(lambda_=training_args.ln_to_weight["query_reg"], 
                                                            T=training_args.max_steps // 3)}
    print("lambda for doc_reg = {}, for query_reg = {}, T = {}".format(
        reg_to_reg_scheduler["doc_reg"].lambda_, reg_to_reg_scheduler["query_reg"].lambda_, reg_to_reg_scheduler["doc_reg"].T
    ))
    print("model: ", args.model_name_or_path, model_type)
    
    trainer = SparseTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=train_collator,
        args=training_args,
        reg_to_reg_scheduler=reg_to_reg_scheduler
    )
    
    if trainer.args.local_rank <= 0:  # only on main process
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.run_name)

        # let's save tokenizer first 
        tokenizer.save_pretrained(args.output_dir)
    
    trainer.train()
    if trainer.is_fsdp_enabled:
        trainer.save_model()
    else:
        if trainer.args.local_rank <= 0:
            trainer.save_model()
            