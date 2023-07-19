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

r"""
Conversion script to convert Huggingface LLaMA checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_hf_llama_to_nemo.py \
     --in-file <path_to_hf_checkpoints_folder> \
     --out-file <path_to_output_nemo_file>
"""

import importlib
import os
import pathlib
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from transformers import LlamaForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel

# from nemo.collections.nlp.models.language_modeling.megatron_llama_model import MegatronLLAMAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed
from nemo.utils.model_utils import inject_model_parallel_rank, uninject_model_parallel_rank


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to Huggingface LLaMA checkpoints",
    )
    parser.add_argument("--out-file", type=str, default=None, required=False, help="Path to output .nemo file.")

    args = parser.parse_args()
    return args


def load_model(cls, checkpoint, strict, **kwargs):
    # print(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
    try:
        if 'cfg' in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            model = ptl_load_state(
                cls, checkpoint, strict=strict, cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg, **kwargs
            )
            # register the artifacts
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg
            if cfg.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_config(llama_config):
    nemo_config = {}
    nemo_config['cfg'] = {}
    nemo_config['cfg']['encoder_seq_length'] = llama_config['max_sequence_length']
    nemo_config['cfg']['num_layers'] = int(llama_config['num_hidden_layers'])
    nemo_config['cfg']['hidden_size'] = llama_config['hidden_size']
    nemo_config['cfg']['ffn_hidden_size'] = llama_config['intermediate_size']
    nemo_config['cfg']['num_attention_heads'] = llama_config['num_attention_heads']
    nemo_config['cfg']['max_position_embeddings'] = llama_config['max_position_embeddings']
    nemo_config['cfg']['init_method_std'] = llama_config['initializer_range']
    nemo_config['cfg']['normalization'] = 'rmsnorm'
    nemo_config['cfg']['layernorm_epsilon'] = llama_config['rms_norm_eps']
    nemo_config['cfg']['pre_process'] = True
    nemo_config['cfg']['post_process'] = True
    nemo_config['cfg']['bias'] = False
    nemo_config['cfg']['hidden_dropout'] = 0.0
    nemo_config['cfg']['attention_dropout'] = 0.0
    nemo_config['cfg']['ffn_dropout'] = 0.0
    nemo_config['cfg']['bias_dropout_add_fusion'] = False
    nemo_config['cfg']['bias_activation_fusion'] = False
    nemo_config['cfg']['use_cpu_initialization'] = False
    nemo_config['cfg']['share_embeddings_and_output_weights'] = False
    nemo_config['cfg']['make_vocab_size_divisible_by'] = 128
    nemo_config['cfg']['activation'] = 'swiglu'
    nemo_config['cfg']['transformer_block_type'] = 'pre_ln'
    nemo_config['cfg']['position_embedding_type'] = 'rope'
    nemo_config['cfg']['precision'] = 32
    nemo_config['cfg']['optim'] = {'name': 'fused_adam'}
    nemo_config['cfg']['tokenizer'] = {}
    nemo_config['cfg']['tokenizer']['library'] = 'sentencepiece'
    nemo_config['cfg']['tokenizer']['type'] = 'null'
    nemo_config['cfg']['tokenizer']['model'] = 'tokenizer.model'
    nemo_config['cfg']['tokenizer']['vocab_file'] = 'null'
    nemo_config['cfg']['tokenizer']['merge_file'] = 'null'
    nemo_config['cfg']['tokenizer']['tokenizer_model'] = 'null'
    nemo_config['cfg']['tokenizer']['sentencepiece_legacy'] = False
    nemo_config['cfg']['micro_batch_size'] = 1
    nemo_config['cfg']['global_batch_size'] = 1

    nemo_config['cfg']['use_scaled_init_method'] = True
    nemo_config['cfg']['normalize_attention_scores'] = True
    nemo_config['cfg']['grad_allreduce_chunk_size_mb'] = 125
    nemo_config['cfg']['persist_layer_norm'] = True
    nemo_config['cfg']['masked_softmax_fusion'] = True
    # print(nemo_config)
    return nemo_config


def convert(args):

    # trainer = Trainer(devices=args.gpus_per_node, accelerator='cpu', num_nodes=num_nodes)
    trainer = Trainer(accelerator='cpu')
    # checkpoint_path = megatron_lm_inject_model_parallel_rank(
    #    os.path.join(args.checkpoint_folder, args.checkpoint_name)
    # )
    logging.info(f"loading checkpoint {args.in_file}")
    model = LlamaForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    nemo_config = load_config(hf_config)
    print(f"hf_config: {hf_config}")
    print(f"nemo_config: {nemo_config}")

    # print("named parameters:")
    # for name, param in model.named_parameters():
    #    print(f"- {name}")

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]

    # print(model)
    # print(model.state_dict())

    checkpoint = None
    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    embed_weight = model.state_dict()[f'model.embed_tokens.weight']
    embed_weights_base_name = f'model.language_model.embedding.word_embeddings.weight'
    checkpoint['state_dict'][embed_weights_base_name] = embed_weight

    rotary_embed_weight = model.state_dict()[f'model.layers.0.self_attn.rotary_emb.inv_freq']
    rotary_embed_weight_base_name = f'model.language_model.rotary_pos_emb.inv_freq'
    checkpoint['state_dict'][rotary_embed_weight_base_name] = rotary_embed_weight

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        old_tensor_shape = model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'].size()
        new_tensor_shape = (head_num, head_size) + model.state_dict()[
            f'model.layers.{l}.self_attn.q_proj.weight'
        ].size()[1:]
        q = model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'].view(*new_tensor_shape)
        k = model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight'].view(*new_tensor_shape)
        v = model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight'].view(*new_tensor_shape)
        qkv_weights = torch.cat((q, k, v), axis=1)
        qkv_weights = qkv_weights.reshape([3 * hidden_size, hidden_size])
        qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
        checkpoint['state_dict'][qkv_weights_base_name] = qkv_weights

        # attention dense
        o_weight = model.state_dict()[f'model.layers.{l}.self_attn.o_proj.weight']
        o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        checkpoint['state_dict'][o_weight_base_name] = o_weight

        # MLP
        mlp_down_weight = model.state_dict()[f'model.layers.{l}.mlp.gate_proj.weight']
        mlp_down_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
        checkpoint['state_dict'][mlp_down_base_name] = mlp_down_weight

        mlp_gate_weight = model.state_dict()[f'model.layers.{l}.mlp.up_proj.weight']
        mlp_gate_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h_2.weight'
        checkpoint['state_dict'][mlp_gate_base_name] = mlp_gate_weight

        mlp_up_weight = model.state_dict()[f'model.layers.{l}.mlp.down_proj.weight']
        mlp_up_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
        checkpoint['state_dict'][mlp_up_base_name] = mlp_up_weight

        # LayerNorm
        input_ln_weight = model.state_dict()[f'model.layers.{l}.input_layernorm.weight']
        input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = input_ln_weight

        post_attn_ln_weight = model.state_dict()[f'model.layers.{l}.post_attention_layernorm.weight']
        post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = post_attn_ln_weight

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'model.norm.weight']
    final_ln_base_name = f'model.language_model.encoder.final_layernorm.weight'
    checkpoint['state_dict'][final_ln_base_name] = final_ln_weight

    output_layer_weight = model.state_dict()[f'lm_head.weight']
    output_layer_base_name = f'model.language_model.output_layer.weight'
    checkpoint['state_dict'][output_layer_base_name] = output_layer_weight

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = OmegaConf.create(nemo_config)

    model = load_model(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)

    # verify tensor parallel rank id and pipeline parallel rank id matches
    model._save_restore_connector = NLPSaveRestoreConnector()
    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == '__main__':
    args = get_args()
    os.chdir(args.in_file)
    convert(args)
