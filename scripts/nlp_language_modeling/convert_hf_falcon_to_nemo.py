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
```
"""

import os
import logging
import time
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, FalconConfig


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
# [ ] test on non parallel attention model （block by no alibi support?) 
# [Y] hf config name mapping for falcon 7b and 40b.
# [Y] trust remote code add
# [ ] MQA MHA GQA num_kv_heads and mcore's GQA logic add (not sure about MQA)
# [ ] When bias_gelu_fusion is True, add_bias_linear must also be True. error
# [ ] remove unnecessary comments and codes.

def setup_logging(log_file="test_log.txt"):
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to Huggingface Falcon checkpoints",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument("--tokenizer-type", type=str, default="tiiuae/falcon-7b", help="Tokenizer type to use, e.g., 'tiiuae/falcon-7b'."
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
                    print(f"Unexpected key: {name} not in checkpoint but in model.")
            if len(checkpoint['state_dict'].keys()) != 0:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )

            # register the artifacts
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
            if cfg.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_falcon_config(args) -> FalconConfig:
    """ Helper utility to load FalconConfig.

    7B and 40B are not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. need to manually set the config values
    and force to `falcon` model type. 
    """
    config = FalconConfig.from_pretrained(args.in_file)

    if config.model_type == 'RefinedWeb':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": config.n_head_kv,
            "new_decoder_architecture": True
        }
    elif config.model_type == 'RefinedWebModel':
        mappings = {
            "num_hidden_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "num_kv_heads": 1 if config.multi_query else config.n_head, 
            "new_decoder_architecture": False
        }
    else:
        return config

    for key, value in mappings.items():
        setattr(config, key, value)

    config.model_type = 'falcon'
    return config


def load_config(args):
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
    if falcon_config.alibi:
        raise ValueError("Alibi is not yet supported in Megatron Core")
    else:
        nemo_config.position_embedding_type = 'rope'
    nemo_config.bias = falcon_config.bias
    nemo_config.hidden_dropout = falcon_config.hidden_dropout
    nemo_config.attention_dropout = falcon_config.attention_dropout
    # need to map to nemo config as well. 
    tokenizer_dict = {
        'library': 'huggingface',
        'type': args.tokenizer_type,  # FIXME: can it work from local args.input too, fix for falcon family?
    }
    
    nemo_config.tokenizer = tokenizer_dict
    ##############################################
    # need refactor Mcore to support parallel attn
    #nemo_config.new_decoder_architecture = falcon_config['new_decoder_architecture'] #bool, if True, always use parallel attn
    #nemo_config.parallel_attention = falcon_config['parallel_attn']
    ###############################################
    #if hasattr(falcon_config,'num_kv_heads'):
    if falcon_config.new_decoder_architecture or falcon_config.multi_query:
        nemo_config.num_query_groups = falcon_config.num_kv_heads
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = 'gelu'
    if falcon_config.rope_scaling is not None:
        if falcon_config.rope_scaling.type == 'linear':
            nemo_config['seq_len_interpolation_factor'] = falcon_config.rope_scaling.factor
        else:
            raise ValueError("Only linear rope scaling type is supported now")

    base = 128
    while falcon_config.vocab_size % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.in_file}")
    tik = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.in_file, trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained(args.in_file)
    hf_config = load_falcon_config(args)
      
    # print(f"hf_config: {hf_config}")
    # print("named parameters:")
    # for name, param in model.named_parameters():
    #     print(f"- {name}")
    
    # add debug state dict list
    hf_keys = list(model.state_dict().keys())

    nemo_config = load_config(args)

    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.get('native_amp_init_scale', 2 ** 32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision == ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback

    nemo_config.precision = precision

    trainer = Trainer(plugins=plugins, accelerator='cpu', precision=precision)

    hidden_size = hf_config.hidden_size
    head_num = hf_config.num_attention_heads
    head_size = hidden_size // head_num 
    num_layers = hf_config.num_hidden_layers
    
    nemo_config.mcore_gpt = True
    nemo_config.transformer_engine = True
    logging.info(f"nemo_config {nemo_config}")
    logging.info(f"mcore_gpt: {nemo_config.mcore_gpt}")
    logging.info(f"transformer_engine: {nemo_config.transformer_engine}")
    # assert nemo_config.mcore_gpt == nemo_config.get(
    #     'transformer_engine', False
    # ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()
    
    def add_to_checkpoint(source_prefix, target_prefix, weight_or_bias):
        source_name = f"{source_prefix}.{weight_or_bias}"
        if source_name in model.state_dict():
            target_name = f"{target_prefix}.{weight_or_bias}"
            checkpoint['state_dict'][target_name] = param_to_weights(model.state_dict()[source_name])
            
            # add debug remove mapped keys
            if source_name in hf_keys:
                hf_keys.remove(source_name)

    def add_weight_and_possible_bias(source_prefix, target_prefix):
        add_to_checkpoint(source_prefix, target_prefix, 'weight')
        if f"{source_prefix}.bias" in model.state_dict():
            add_to_checkpoint(source_prefix, target_prefix, 'bias')
    
    add_to_checkpoint('transformer.word_embeddings', 'model.embedding.word_embeddings', 'weight')

    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, f'head_num ({head_num}) must be divisible by num_query_groups ({num_query_groups})'

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        prefix = f'transformer.h.{l}'
        

        # HF: [num_heads x 3 x head_dim, hidden_size], interleaved qkv weights
        # Query types and expected kv heads.
        #  - MHA: num_heads = num_kv_heads
        #  - Multi-Query Attention: num_kv_heads = 1
        #  - Grouped-Query Attention: num_heads % num_kv_heads = 0

        add_weight_and_possible_bias(f'{prefix}.self_attention.query_key_value', f'model.decoder.layers.{l}.self_attention.linear_qkv')
        add_weight_and_possible_bias(f'{prefix}.self_attention.dense', f'model.decoder.layers.{l}.self_attention.linear_proj')
        add_weight_and_possible_bias(f'{prefix}.mlp.dense_h_to_4h', f'model.decoder.layers.{l}.mlp.linear_fc1')
        add_weight_and_possible_bias(f'{prefix}.mlp.dense_4h_to_h', f'model.decoder.layers.{l}.mlp.linear_fc2')

        if hf_config.new_decoder_architecture:
            add_weight_and_possible_bias(f'{prefix}.ln_attn', f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm')
            add_weight_and_possible_bias(f'{prefix}.ln_mlp', f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm')
        else:
            add_weight_and_possible_bias(f'{prefix}.input_layernorm', f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm')
            if not hf_config.parallel_attn:
                add_weight_and_possible_bias(f'{prefix}.post_attention_layernorm', f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm')

        print(f"done layer {l}")

    # final layer norm
    add_weight_and_possible_bias('transformer.ln_f', 'model.decoder.final_layernorm')

    # LM weight
    add_to_checkpoint('lm_head', 'model.output_layer','weight')
    
    if hf_keys:
        logging.warning(f"Some keys in HuggingFace's model didn't get mapped to NeMo's state_dict: {hf_keys}")
    
    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logging.info(f'Weights loaded. Total time: {t}')

    del model

    model = load_model(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == '__main__':
    setup_logging()
    args = get_args()
    convert(args)