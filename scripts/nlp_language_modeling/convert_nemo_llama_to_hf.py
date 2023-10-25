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

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging

"""
    This script will generate a .bin file which can be converted back to hf format using the below code.
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    LLAMA_PRETRAINED_PATH = "/home/llama-2-7B-hf/"
    NEMO_EXPORTED_PATH = "/home/nemo-exported.bin"

    model = AutoModelForCausalLM.from_pretrained(LLAMA_PRETRAINED_PATH, local_files_only=True)
    nemo_exported = torch.load(NEMO_EXPORTED_PATH)

    model.load_state_dict(nemo_exported['state_dict'])
    model.save_pretrained("./nemo-exported-llama/")

    Known constraints, this script:
        1. is tested on 7B and 13B model only; 70B (GQA) support will be added soon.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to .nemo file",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    args = parser.parse_args()
    return args


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    map_location = torch.device('cpu') if cpu_only else None
    model = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, map_location=map_location)
    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    hidden_size = model.cfg.hidden_size
    head_num = model.cfg.num_attention_heads
    num_layers = model.cfg.num_layers
    ffn_hidden_size = model.cfg.ffn_hidden_size
    head_size = hidden_size // head_num

    # Embedding
    embed_weight = model.state_dict()[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(embed_weight)

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        q_weight = torch.empty(head_num * head_size, hidden_size)
        k_weight = torch.empty(head_num * head_size, hidden_size)
        v_weight = torch.empty(head_num * head_size, hidden_size)

        idx = 0
        while head_size * idx < hidden_size:
            q_weight[head_size * idx : head_size * (idx + 1), :] = qkv_weights[
                idx * (3 * head_size) : idx * (3 * head_size) + head_size, :
            ]
            k_weight[head_size * idx : head_size * (idx + 1), :] = qkv_weights[
                idx * (3 * head_size) + head_size : idx * (3 * head_size) + (2 * head_size), :
            ]
            v_weight[head_size * idx : head_size * (idx + 1), :] = qkv_weights[
                idx * (3 * head_size) + (2 * head_size) : idx * (3 * head_size) + (3 * head_size), :
            ]
            idx += 1

        q_weights_base_name = f'model.layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'model.layers.{l}.self_attn.k_proj.weight'
        v_weights_base_name = f'model.layers.{l}.self_attn.v_proj.weight'

        checkpoint['state_dict'][q_weights_base_name] = param_to_weights(q_weight)
        checkpoint['state_dict'][k_weights_base_name] = param_to_weights(k_weight)
        checkpoint['state_dict'][v_weights_base_name] = param_to_weights(v_weight)

        # attention dense
        o_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj.weight']
        o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
        checkpoint['state_dict'][o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight'

        checkpoint['state_dict'][mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint['state_dict'][mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2.weight']
        mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        checkpoint['state_dict'][mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'model.decoder.final_layernorm.weight']
    final_ln_base_name = f'model.norm.weight'
    checkpoint['state_dict'][final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()[f'model.output_layer.weight']
    output_layer_base_name = f'lm_head.weight'
    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)


if __name__ == '__main__':
    args = get_args()
    input_nemo_file = args.in_file
    output_hf_file = args.out_file
    convert(input_nemo_file, output_hf_file, precision=args.precision)
