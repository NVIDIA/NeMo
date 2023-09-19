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

"""
Conversion script to convert Huggingface Falcon 1B/7B/40B/180B checkpoints into nemo checkpoint.

This script will generate a Megatron model with TP=1 and PP=1. If you need different TP/PP
values, then after running this script, please use the script located below to set the
TP/PP values you want:
    NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py
    
Example to run this conversion script:
```
    python convert_hf_falcon_to_nemo.py \
     --in-file <path_to_hf_checkpoints_folder> \
     --out-file <path_to_output_nemo_file> \
     --tokenizer-type <model_id on hf> \
     --precision <precision of converted nemo model>
```
"""

import logging
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)

# TODO:
# [Y] refactor ckpt func to make it cleaner
# [Y] dict tokenizer mapping for falcon family
# [ ] good way to add new_decoder_architecture and parallel_attn in megatron_gpt_config.yaml
# [ ] safetensors loading. (only 180b used safetensors)
# [Y] test on non parallel attention model （block by no alibi support? 1b-rw good, 7b-rw still some time)
# [Y] hf config name mapping for falcon 7b and 40b.
# [Y] trust remote code add
# [Y] MQA MHA GQA num_kv_heads and mcore's GQA logic add (not sure about MQA)
# [Y] When bias_gelu_fusion is True, add_bias_linear must also be True. error
# [ ] remove unnecessary comments and codes.


def setup_logging(log_file="test_log.txt"):
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
    )


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to Huggingface Falcon checkpoints",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="tiiuae/falcon-7b",
        help="Tokenizer type to use, e.g., 'tiiuae/falcon-7b'.",
    )
    args = parser.parse_args()
    return args


def load_model(cls, checkpoint, strict, **kwargs):
    try:
        if 'cfg' in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            model = cls(cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], **kwargs)
            for name, module in model.named_parameters():
                if name in checkpoint['state_dict']:
                    module.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)
                else:
                    logging.info(f"Unexpected key: {name} not in checkpoint but in model.")
            if len(checkpoint['state_dict'].keys()) != 0:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_falcon_config(args) -> FalconConfig:
    """ Helper utility to load FalconConfig.

    Falcon-7B and Falcon-40B are not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. need to manually set the config values
    and force to `falcon` model type. 
    """
    config = FalconConfig.from_pretrained(args.in_file)

    if config.model_type == 'RefinedWeb':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": config.n_head_kv,
            "new_decoder_architecture": True,
        }
    elif config.model_type == 'RefinedWebModel':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": 1 if config.multi_query else config.n_head,
            "new_decoder_architecture": False,
        }
    else:
        return config

    for key, value in mappings.items():
        setattr(config, key, value)

    config.model_type = 'falcon'
    return config


def load_nemo_config(args):
    falcon_config = load_falcon_config(args)
    logging.info(f"falcon_config, {falcon_config}")
    nemo_config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_gpt_config.yaml')
    ).model
    nemo_config.encoder_seq_length = falcon_config.max_position_embeddings
    nemo_config.num_layers = int(falcon_config.num_hidden_layers)
    nemo_config.hidden_size = falcon_config.hidden_size
    nemo_config.num_attention_heads = falcon_config.num_attention_heads
    nemo_config.max_position_embeddings = falcon_config.max_position_embeddings
    nemo_config.init_method_std = falcon_config.initializer_range
    nemo_config.layernorm_epsilon = falcon_config.layer_norm_epsilon
    try:
        if falcon_config.alibi:
            raise ValueError(
                "Alibi is not yet supported in Megatron Core, \
                force to use RoPE will generate suboptimal responses"
            )
    except ValueError as e:
        print(e)
    finally:
        nemo_config.position_embedding_type = 'rope'
    nemo_config.bias = falcon_config.bias
    nemo_config.hidden_dropout = falcon_config.hidden_dropout
    nemo_config.attention_dropout = falcon_config.attention_dropout
    # TODO: how does vocab_file, merge_file etc get mapped automatically in respect to variants of falcon models?
    tokenizer_dict = {
        'library': 'huggingface',
        'type': args.tokenizer_type,  # FIXME: can it work from local args.input too, fix for falcon family?
    }

    nemo_config.tokenizer = tokenizer_dict
    ##############################################
    # TODO: need refactor Mcore to support parallel attn and 40b/180b model arch
    # nemo_config.new_decoder_architecture = falcon_config['new_decoder_architecture'] #bool, if True, always use parallel attn
    # nemo_config.parallel_attention = falcon_config['parallel_attn']
    ###############################################

    nemo_config.num_query_groups = (
        falcon_config.num_kv_heads if falcon_config.new_decoder_architecture or falcon_config.multi_query else None
    )
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'gelu'
    if falcon_config.rope_scaling is not None:
        if falcon_config.rope_scaling.type == 'linear':
            nemo_config['seq_len_interpolation_factor'] = falcon_config.rope_scaling.factor
        else:
            raise ValueError("Only linear rope scaling type is supported now")

    nemo_config.mcore_gpt = True
    nemo_config.transformer_engine = True
    nemo_config.bias_activation_fusion = False
    nemo_config.bias_dropout_add_fusion = False

    base = 128
    while falcon_config.vocab_size % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def determine_precision(args):
    """Helper function to determine the precision of model
    """
    if args.precision in ["32", "16"]:
        return int(args.precision)
    elif args.precision in ["bf16", "bf16-mixed"]:
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            return args.precision[2:]  # prune 'bf' from string
    return args.precision


def determine_dtype(precision):
    dtype_map = {
        "32": torch.float32,
        "16": torch.float16,
        "16-mixed": torch.float16,
        "bf16": torch.bfloat16,
        "bf16-mixed": torch.bfloat16,
    }
    return dtype_map.get(precision, torch.float32)  # default to torch.float32


def convert(args):
    logging.info(f"loading checkpoint {args.in_file}")
    tik = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.in_file, trust_remote_code=True)
    falcon_config = load_falcon_config(args)
    # debug
    logging.debug(f"initial falcon_config, {falcon_config}")

    nemo_config = load_nemo_config(args)
    precision = determine_precision(args)

    plugins = []

    if precision in ['16', '16-mixed', 'bf16', 'bf16-mixed']:
        scaler_params = {
            'init_scale': nemo_config.get('native_amp_init_scale', 2 ** 32),
            'growth_interval': nemo_config.get('native_amp_growth_interval', 1000),
            'hysteresis': nemo_config.get('hysteresis', 2),
        }

        plugin_precision = '16-mixed' if precision in ['16', '16-mixed'] else 'bf16-mixed'
        scaler = GradScaler(**scaler_params) if precision in ['16', '16-mixed'] else None

    dtype = determine_dtype(precision)
    nemo_config.precision = precision
    trainer = Trainer(plugins=plugins, accelerator='cpu', precision=precision)

    hidden_size = falcon_config.hidden_size
    head_num = falcon_config.num_attention_heads
    head_size = hidden_size // head_num
    num_layers = falcon_config.num_hidden_layers

    #  - MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    num_query_groups = (
        nemo_config.num_query_groups
        if nemo_config.num_query_groups and nemo_config.num_query_groups != head_num
        else head_num
    )
    assert (
        head_num % num_query_groups == 0
    ), f'head_num ({head_num}) must be divisible by num_query_groups ({num_query_groups})'

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    def add_to_checkpoint(source_prefix, target_prefix, weight_or_bias, is_layernorm=False):
        source_name = f"{source_prefix}.{weight_or_bias}"
        if source_name in model.state_dict():
            if is_layernorm:
                target_name = f"{target_prefix}_{weight_or_bias}"
            else:
                target_name = f"{target_prefix}.{weight_or_bias}"
            checkpoint['state_dict'][target_name] = param_to_weights(model.state_dict()[source_name])

    def add_weight_and_possible_bias(source_prefix, target_prefix, is_layernorm=False):
        add_to_checkpoint(source_prefix, target_prefix, 'weight', is_layernorm)
        if f"{source_prefix}.bias" in model.state_dict():
            add_to_checkpoint(source_prefix, target_prefix, 'bias', is_layernorm)

    add_to_checkpoint('transformer.word_embeddings', 'model.embedding.word_embeddings', 'weight')

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        prefix = f'transformer.h.{l}'

        add_weight_and_possible_bias(
            f'{prefix}.self_attention.query_key_value', f'model.decoder.layers.{l}.self_attention.linear_qkv'
        )
        add_weight_and_possible_bias(
            f'{prefix}.self_attention.dense', f'model.decoder.layers.{l}.self_attention.linear_proj'
        )
        add_weight_and_possible_bias(f'{prefix}.mlp.dense_h_to_4h', f'model.decoder.layers.{l}.mlp.linear_fc1')
        add_weight_and_possible_bias(f'{prefix}.mlp.dense_4h_to_h', f'model.decoder.layers.{l}.mlp.linear_fc2')

        if falcon_config.new_decoder_architecture:
            add_weight_and_possible_bias(
                f'{prefix}.ln_attn',
                f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm',
                is_layernorm=True,
            )
            add_weight_and_possible_bias(
                f'{prefix}.ln_mlp', f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm', is_layernorm=True
            )
        else:
            add_weight_and_possible_bias(
                f'{prefix}.input_layernorm',
                f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm',
                is_layernorm=True,
            )
            if not falcon_config.parallel_attn:
                add_weight_and_possible_bias(
                    f'{prefix}.post_attention_layernorm',
                    f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm',
                    is_layernorm=True,
                )

        print(f"done layer {l}")

    # final layer norm
    add_weight_and_possible_bias('transformer.ln_f', 'model.decoder.final_layernorm')

    # LM weight
    add_to_checkpoint('lm_head', 'model.output_layer', 'weight')

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    logging.debug(f'final checkpoint, {checkpoint}')

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logging.info(f'Weights loaded. Total time: {t}')

    del model

    # model = load_model(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)
    model = MegatronGPTModel(checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY], trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False
    # We make sure that the tokenizer can be instantiated later regardless of args.input
    model.cfg.tokenizer.update(type=args.tokenizer_type)
    # save model
    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == '__main__':
    setup_logging()
    args = get_args()
    convert(args)
