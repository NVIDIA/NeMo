# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer
from transformers import AutoModelForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

"""
Script to convert a starcoder checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_nemo_starcoder_to_hf.py \
    --in-file /path/to/file.nemo or /path/to/extracted_folder \
    --out-file /path/to/pytorch_model.bin

2) Generate the full HF model folder

    python convert_nemo_starcoder_to_hf.py \
    --in-file /path/to/file.nemo or /path/to/extracted_folder \
    --out-file /path/to/pytorch_model.bin \
    --hf-in-file /path/to/input_hf_folder \
    --hf-out-file /path/to/output_hf_folder

    Use the --cpu-only flag if the model cannot fit in the GPU. 
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, required=False, help="Path to .nemo file",
    )
    parser.add_argument(
        "--in-ckpt", type=str, required=False, help="Path to .ckpt file OR folder of distributed checkpoint",
    )
    parser.add_argument("--out-file", type=str, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--hf-in-path",
        type=str,
        default=None,
        help="A HF model path, "
             "e.g. a folder containing hhttps://huggingface.co/bigcode/starcoder/tree/main",
    )
    parser.add_argument(
        "--hf-out-path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
             "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
             "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    if cpu_only:
        map_location = torch.device('cpu')
        model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
        model_config.use_cpu_initialization = True
        model_config.tensor_model_parallel_size = 1
    else:
        map_location, model_config = None, None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )
    if precision is None:
        precision = model.cfg.precision
    try:
        dtype = torch_dtype_from_precision(precision)
    except ValueError as e:
        logging.warning(str(e) + f", precision string '{precision}' is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    def get_original_key(new_key):
        new_key = new_key[len(prefix):]

        if new_key.startswith("embedding.word_embeddings.weight"):
            return "transformer.wte.weight"
        if new_key.startswith("model.embedding.position_embeddings.weight"):
            return "transformer.wpe.weight"

        key = new_key.replace("decoder.layers", "transformer.h")

        key = key.replace("self_attention.linear_proj", "attn.c_proj")
        key = key.replace("self_attention.linear_qkv.layer_norm_", "ln_1.")
        key = key.replace("self_attention.linear_qkv", "attn.c_attn")
        key = key.replace("mlp.linear_fc1.layer_norm_", "ln_2.")
        key = key.replace("linear_fc1", "c_fc")
        key = key.replace("linear_fc2", "c_proj")
        return key

    prefix = 'model.module.' if any(k.startswith('model.module.') for k in model.state_dict()) else 'model.'

    for key, value in model.state_dict().items():
        if '_extra_state' in key:
            continue
        orig_key = get_original_key(key)
        checkpoint[orig_key] = param_to_weights(value)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights reverted and saved to {output_hf_file}")


def replace_hf_weights(weights_file, input_hf_path, output_hf_path):
    model = AutoModelForCausalLM.from_pretrained(input_hf_path, local_files_only=True)
    nemo_exported = torch.load(weights_file)

    model.load_state_dict(nemo_exported)
    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")


if __name__ == '__main__':
    args = get_args()
    assert args.in_file or args.in_ckpt

    convert(args.in_file, args.out_file, precision=args.precision, cpu_only=args.cpu_only)
    if args.hf_in_path and args.hf_out_path:
        replace_hf_weights(args.out_file, args.hf_in_path, args.hf_out_path)
    else:
        logging.info("`hf-in-path` and/or `hf-out-path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.out_file}")