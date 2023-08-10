# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
Conversion script to convert Huggingface checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_hf_bloom_to_nemo.py \
     --in-file <path_to_HF_checkpoints_folder> \
     --out-file <path_to_output_nemo_file>
"""

from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import BloomForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file",
        type=str,
        default="/dataset/bloom-7b1/",
        required=False,
        help="Path to HF BLOOM checkpoints saved during training. Ex: /dataset/bloom-560m/",
    )
    parser.add_argument(
        "--out-file", type=str, default='bloom.nemo', required=False, help="Path to output .nemo file."
    )

    args = parser.parse_args()
    return args


def load_model(cls, checkpoint, strict, **kwargs):
    print(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
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


def load_config(hf_config):
    nemo_config = {}
    nemo_config['cfg'] = {}
    nemo_config['cfg']['encoder_seq_length'] = 2048
    nemo_config['cfg']['num_layers'] = hf_config['n_layer']
    nemo_config['cfg']['hidden_size'] = hf_config['hidden_size']
    nemo_config['cfg']['ffn_hidden_size'] = 4 * hf_config['hidden_size']
    nemo_config['cfg']['num_attention_heads'] = hf_config['n_head']
    nemo_config['cfg']['max_position_embeddings'] = 2048
    nemo_config['cfg']['make_vocab_size_divisible_by'] = 256  # must be set
    nemo_config['cfg']['init_method_std'] = hf_config['initializer_range']
    nemo_config['cfg']['normalization'] = 'layernorm'
    nemo_config['cfg']['embedding_normalization'] = 'layernorm'
    nemo_config['cfg']['layernorm_epsilon'] = hf_config['layer_norm_epsilon']
    nemo_config['cfg']['attention_dropout'] = hf_config['attention_dropout']
    nemo_config['cfg']['hidden_dropout'] = hf_config['hidden_dropout']
    nemo_config['cfg']['pre_process'] = True
    nemo_config['cfg']['post_process'] = True
    nemo_config['cfg']['bias'] = True
    nemo_config['cfg']['bias_dropout_add_fusion'] = True
    nemo_config['cfg']['masked_softmax_fusion'] = hf_config['masked_softmax_fusion']
    nemo_config['cfg']['bias_activation_fusion'] = True
    nemo_config['cfg']['share_embeddings_and_output_weights'] = False
    nemo_config['cfg']['apply_query_key_layer_scaling'] = False
    nemo_config['cfg']['activation'] = 'gelu'
    nemo_config['cfg']['transformer_block_type'] = 'pre_ln'
    nemo_config['cfg']['position_embedding_type'] = 'alibi'
    nemo_config['cfg']['precision'] = 16
    nemo_config['cfg']['optim'] = {'name': 'fused_adam'}
    nemo_config['cfg']['tokenizer'] = {}
    nemo_config['cfg']['tokenizer']['library'] = 'huggingface'
    nemo_config['cfg']['tokenizer']['type'] = hf_config['_name_or_path']
    nemo_config['cfg']['tokenizer']['model'] = 'null'
    nemo_config['cfg']['tokenizer']['vocab_file'] = 'null'
    nemo_config['cfg']['tokenizer']['merge_file'] = 'null'
    nemo_config['cfg']['micro_batch_size'] = 1
    nemo_config['cfg']['global_batch_size'] = 1

    return nemo_config


def convert(args):
    trainer = Trainer(devices=1, accelerator='cpu', num_nodes=1)

    logging.info(f"loading checkpoint {args.in_file}")
    model = BloomForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    nemo_config = load_config(hf_config)

    num_layers = hf_config["n_layer"]
    hf_state_dict = model.state_dict()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    # embedding
    checkpoint['state_dict'][f'model.language_model.embedding.word_embeddings.weight'] = hf_state_dict[
        f'transformer.word_embeddings.weight'
    ]

    # input layernorm
    input_ln_weight = hf_state_dict[f'transformer.word_embeddings_layernorm.weight']
    checkpoint['state_dict'][f'model.language_model.embedding.embedding_layernorm.weight'] = input_ln_weight

    input_ln_bias = hf_state_dict[f'transformer.word_embeddings_layernorm.bias']
    checkpoint['state_dict'][f'model.language_model.embedding.embedding_layernorm.bias'] = input_ln_bias

    for l in range(num_layers):
        print(f"converting layer {l}")

        # self_attention
        qkv_weights = hf_state_dict[f'transformer.h.{l}.self_attention.query_key_value.weight']

        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
        ] = qkv_weights

        qkv_bias = hf_state_dict[f'transformer.h.{l}.self_attention.query_key_value.bias']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.bias'
        ] = qkv_bias

        # attention dense
        dense_weight = hf_state_dict[f'transformer.h.{l}.self_attention.dense.weight']
        checkpoint['state_dict'][f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'] = dense_weight

        dense_bias = hf_state_dict[f'transformer.h.{l}.self_attention.dense.bias']
        checkpoint['state_dict'][f'model.language_model.encoder.layers.{l}.self_attention.dense.bias'] = dense_bias

        # MLP
        dense_h_to_4h_weight = hf_state_dict[f'transformer.h.{l}.mlp.dense_h_to_4h.weight']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
        ] = dense_h_to_4h_weight

        dense_h_to_4h_bias = hf_state_dict[f'transformer.h.{l}.mlp.dense_h_to_4h.bias']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.bias'
        ] = dense_h_to_4h_bias

        dense_4h_to_h_weight = hf_state_dict[f'transformer.h.{l}.mlp.dense_4h_to_h.weight']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
        ] = dense_4h_to_h_weight

        dense_4h_to_h_bias = hf_state_dict[f'transformer.h.{l}.mlp.dense_4h_to_h.bias']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.bias'
        ] = dense_4h_to_h_bias

        # input LayerNorm
        input_ln_weight = hf_state_dict[f'transformer.h.{l}.input_layernorm.weight']
        checkpoint['state_dict'][f'model.language_model.encoder.layers.{l}.input_layernorm.weight'] = input_ln_weight

        input_ln_bias = hf_state_dict[f'transformer.h.{l}.input_layernorm.bias']
        checkpoint['state_dict'][f'model.language_model.encoder.layers.{l}.input_layernorm.bias'] = input_ln_bias

        # post attention layernorm
        post_attn_ln_weight = hf_state_dict[f'transformer.h.{l}.post_attention_layernorm.weight']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        ] = post_attn_ln_weight

        post_attn_ln_bias = hf_state_dict[f'transformer.h.{l}.post_attention_layernorm.bias']
        checkpoint['state_dict'][
            f'model.language_model.encoder.layers.{l}.post_attention_layernorm.bias'
        ] = post_attn_ln_bias

        print(f"done layer {l}")

    # final layernorm
    final_ln_weight = hf_state_dict[f'transformer.ln_f.weight']
    checkpoint['state_dict'][f'model.language_model.encoder.final_layernorm.weight'] = final_ln_weight

    final_ln_bias = hf_state_dict[f'transformer.ln_f.bias']
    checkpoint['state_dict'][f'model.language_model.encoder.final_layernorm.bias'] = final_ln_bias

    # output layer
    output_layer_weight = hf_state_dict[f'lm_head.weight']
    checkpoint['state_dict'][f'model.language_model.output_layer.weight'] = output_layer_weight

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = OmegaConf.create(nemo_config)

    model = load_model(MegatronGPTModel, checkpoint, strict=True, trainer=trainer)
    model = model.to(torch.float16)

    model._save_restore_connector = NLPSaveRestoreConnector()
    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
