import argparse
import os 
import ujson 
from safetensors.torch import save_file, load_file
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    config = LoraConfig.from_pretrained(args.input_dir)
    state_dict = load_file(os.path.join(args.input_dir, "adapter_model.safetensors"))
    
    new_state_dict = {}
    old_prefix = "base_model.model.model"
    new_prefix = "base_model.model"
    for k, v in state_dict.items():
        k = new_prefix + k[len(old_prefix):]
        new_state_dict[k] = v
    
    if config.auto_mapping["base_model_class"] == "LlamaBiForMNTP":
        config.auto_mapping["base_model_class"] = "LlamaBiModel"
    elif config.auto_mapping["base_model_class"] == "Qwen2BiForMNTP":
        config.auto_mapping["base_model_class"] = "Qwen2BiModel"
    else:
        raise ValueError("Unknown base_model_class: {}".format(config.auto_mapping["base_model_class"]))
    os.makedirs(args.output_dir, exist_ok=True)
    save_file(new_state_dict, os.path.join(args.output_dir, "adapter_model.safetensors"))
    config.save_pretrained(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)