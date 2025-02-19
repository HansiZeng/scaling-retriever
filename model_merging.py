import os 
import argparse

import torch
from peft import PeftModel, LoraConfig
from modeling.bidirectional_llama import LlamaBiForMNTP


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sparse_lora_name_or_path", type=str, required=True)
    parser.add_argument("--dense_lora_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--weights", type=str, default='[0.5,0.5]')
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--method", type=str, default='linear')
    
    return parser.parse_args()

def load_lora_model(base_model_name_or_path, 
                    lora_name_or_path,
                    is_trainable=False):
    """
    Load a LoRA model based on the specified base model and LoRA weights.
    """
    base_model = LlamaBiForMNTP.from_pretrained(base_model_name_or_path)
    lora_config = LoraConfig.from_pretrained(lora_name_or_path)
    lora_model = PeftModel.from_pretrained(base_model, 
                                            lora_name_or_path, 
                                            config=lora_config,
                                            is_trainable=is_trainable)
    return lora_model


def check_compatibility(args):
    sparse_config = LoraConfig.from_pretrained(args.sparse_lora_name_or_path)
    dense_config = LoraConfig.from_pretrained(args.dense_lora_name_or_path)
    
    assert sparse_config.base_model_name_or_path == dense_config.base_model_name_or_path, \
        (sparse_config.base_model_name_or_path, dense_config.base_model_name_or_path)
        
    assert sparse_config.auto_mapping["base_model_class"] == dense_config.auto_mapping["base_model_class"], \
        (sparse_config.auto_mapping["base_model_class"], dense_config.auto_mapping["base_model_class"])

if __name__ == '__main__':
    args = parse_args()
    check_compatibility(args)
    
    sparse_config = LoraConfig.from_pretrained(args.sparse_lora_name_or_path)
    base_model = LlamaBiForMNTP.from_pretrained(sparse_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.sparse_lora_name_or_path, adapter_name="sparse")
    _ = model.load_adapter(args.dense_lora_name_or_path, adapter_name="dense")
    
    weights = eval(args.weights)
    print("weights: ", weights)
    if args.method == "linear":
        model.add_weighted_adapter(
            adapters=["sparse", "dense"],
            weights=weights,
            combination_type="linear",
            adapter_name="merge"
        )
        assert weights[0] == weights[1] == 0.5, weights
    elif args.method == "cat":
        model.add_weighted_adapter(
            adapters=["sparse", "dense"],
            weights=weights,
            combination_type="cat",
            adapter_name="merge"
        )
    elif args.method in ["ties", "dare_ties"]:
        model.add_weighted_adapter(
            adapters=["sparse", "dense"],
            weights=weights,
            combination_type=args.method,
            density=args.density,
            adapter_name="merge"
        )
    else:
        raise NotImplementedError(f"Unknown combination method: {args.method}")
    
    output_dir = args.output_dir[:-1] if args.output_dir.endswith("/") else args.output_dir
    output_dir = f"{output_dir}"
    model.set_adapter("merge")
    model.save_pretrained(output_dir)
