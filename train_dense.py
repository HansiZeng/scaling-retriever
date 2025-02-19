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
    LlamaDenseCollatorForNCE,
    LlamaDenseCollatorForMarginMSE,
    LlamaDenseCollatorForKLDiv,
    LlamaDenseCollatorForNCE_KLDiv
)
    
from modeling.llm_encoder import (
    LlamaBiDenseForMarginMSE, 
    LlamaBiDense,
    LlamaBiDenseForKLDiv,
    LlamaBiDenseForNCE_KLDiv,
)
from tasks.dense_trainer import DenseTrainingArgs, DenseTrainer, DenseTrainerForNCE_KLdiv
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
                            is_trainable=True,
                            T=args.T)
    return model

if __name__ == "__main__":
    parser = HfArgumentParser((DenseTrainingArgs))
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
    elif args.loss_type in ["kldiv", "nce_kldiv"]:
        train_dataset = DualEncoderDatasetForKLDiv(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
            n_negs=args.n_negs
        )
    
    if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
        # It is hacky here and inconsistent with train_splade.py
        # As we transform the lora_model's state_dict and config from MNTP to BiModel
        # in the new folder without copy the tokenizer configs.
        # Hence we will load the tokenizer from the base_model_name_or_path
        with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = ujson.load(f)
        tokenizer = AutoTokenizer.from_pretrained(adapter_config["base_model_name_or_path"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if model_type == "llama":
        if args.train_config is not None:
            with open(args.train_config, "r") as fin:
                config = ujson.load(fin)
        else:
            config = None
        if args.loss_type == "nce":
            train_collator = LlamaDenseCollatorForNCE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiDense)
            else:
                model = LlamaBiDense.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "margin_mse":
            train_collator = LlamaDenseCollatorForMarginMSE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiDenseForMarginMSE)
            else:
                model = LlamaBiDenseForMarginMSE.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "kldiv":
            train_collator = LlamaDenseCollatorForKLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiDenseForKLDiv)
            else:
                model = LlamaBiDenseForKLDiv.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "nce_kldiv":
            train_collator = LlamaDenseCollatorForNCE_KLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiDenseForNCE_KLDiv)
            else:
                model = LlamaBiDenseForNCE_KLDiv.build(args.model_name_or_path, args, config=config)
                
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        #if model_type == "llama":
        #    tokenizer.add_special_tokens({"cls_token": "<|eot_id|>"})
        
        # model might not be trainable after loading from lora 
        model.train()
        
    training_args = args
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    if args.loss_type == "nce_kldiv":
        trainer = DenseTrainerForNCE_KLdiv(
            model=model,
            train_dataset=train_dataset,
            data_collator=train_collator,
            args=training_args,
        )
    else:
        trainer = DenseTrainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=train_collator,
            args=training_args,
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
            