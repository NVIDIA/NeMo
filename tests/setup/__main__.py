import argparse
import os

from .data.create_sample_jsonl import create_sample_jsonl
from .models.create_hf_model import create_hf_model

print("Setup test data and models...")

parser = argparse.ArgumentParser("Setup test data and models.")
parser.add_argument("--data_dir", required=True, help="Root save directory for data")
parser.add_argument("--model_dir", required=True, help="Root save directory for models")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files and directories")
args = parser.parse_args()

print(f"Arguments are: {vars(args)}")

os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)

create_sample_jsonl(
    os.path.join(args.data_dir, "test_quantization", "test.json"),
    args.overwrite,
)

create_hf_model(
    "meta-llama/Llama-2-7b-hf",
    os.path.join(args.model_dir, "tiny_llama2_hf"),
    {"hidden_size": 128, "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 4},
    args.overwrite,
)
print("Setup done.")
