import argparse
from dataclasses import dataclass

from nemo.collections import llm


@dataclass
class Llama3ConfigCI(llm.Llama3Config8B):
    seq_length: int = 2048
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 8


def get_args():
    parser = argparse.ArgumentParser(description='Merge LoRA weights with base LLM')
    parser.add_argument('--lora_checkpoint_path', type=str, help="Path to finetuned LORA checkpoint")
    parser.add_argument('--output_path', type=str, help="Path to save merged checkpoint")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    llm.peft.merge_lora(
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
    )
