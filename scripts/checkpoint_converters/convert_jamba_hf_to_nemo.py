import os
from argparse import ArgumentParser

import torch
from omegaconf.omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_jamba_model import MegatronJambaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

'''
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_jamba_hf_to_nemo.py --output_path /home/ataghibakhsh/forks/full_jamba.nemo
'''


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_jamba_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--input_name_or_path", type=str, default="ai21labs/Jamba-v0.1")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    args = parser.parse_args()
    return args


def convert(args):

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.tokenizer.type = "ai21labs/Jamba-v0.1"
    # nemo_config.model.num_attention_heads=8
    # nemo_config.model.num_query_groups=8
    # nemo_config.model.hidden_size=32
    # nemo_config.model.ffn_hidden_size=112
    # nemo_config.model.num_moe_experts=16

    nemo_config.model.use_cpu_initialization = True
    # print(nemo_config)
    # import sys
    # sys.exit()
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("ai21labs/Jamba-v0.1")
    nemo_config.model.hybrid_override_pattern = "M-MOM-MO*-MOM-MO" * 4

    # config.hidden_size = int(config.hidden_size / 128)
    # config.intermediate_size = int(config.intermediate_size  / 128)
    # config.num_attention_heads = int(config.num_attention_heads/4)
    # config.num_key_value_heads = 8
    # import math
    # config.mamba_dt_rank = math.ceil(config.hidden_size / 16)

    # hf_model = AutoModelForCausalLM.from_config(config)#.to("cuda")
    # import sys
    # sys.exit()

    logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.input_name_or_path, trust_remote_code=True
    )  # , force_download=True)

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()
    nemo_config.model.use_cpu_initialization = True
    nemo_model_from_hf = MegatronJambaModel(nemo_config.model, trainer)
    # print(nemo_model_from_hf.state_dict().keys())
    # import sys
    # sys.exit()
    new_state_dict = {}

    new_state_dict['model.embedding.word_embeddings.weight'] = hf_model.state_dict()['model.embed_tokens.weight']
    new_state_dict['model.decoder.final_norm.weight'] = hf_model.state_dict()['model.final_layernorm.weight']
    new_state_dict['model.output_layer.weight'] = hf_model.state_dict()['lm_head.weight']
    for i, symb in enumerate(nemo_model_from_hf.hybrid_override_pattern):
        hf_jamba_layer = int(i / 2)
        if symb == "M":

            new_state_dict[f'model.decoder.layers.{i}.mixer.A_log'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.A_log'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.D'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.D'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.conv1d.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.bias'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.conv1d.bias'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.in_proj.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.in_proj.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.x_proj.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.x_proj.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.dt_proj.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.dt_proj.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.dt_proj.bias'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.dt_proj.bias'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.out_proj.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.out_proj.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.dt_layernorm.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.dt_layernorm.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.b_layernorm.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.b_layernorm.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mixer.c_layernorm.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.mamba.c_layernorm.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.norm.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.input_layernorm.weight'
            ]
        if symb == "*":

            new_state_dict[f'model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'] = (
                hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.input_layernorm.weight']
            )

            new_state_dict[f'model.decoder.layers.{i}.self_attention.linear_proj.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.self_attn.o_proj.weight'
            ]
            hidden_size = config.hidden_size
            head_num = config.num_attention_heads
            head_size = hidden_size // head_num
            num_query_groups = config.num_key_value_heads

            old_tensor_shape = hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.self_attn.q_proj.weight'].size()
            new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
            new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

            q = hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.self_attn.q_proj.weight'].view(
                *new_q_tensor_shape
            )
            k = hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.self_attn.k_proj.weight'].view(
                *new_kv_tensor_shape
            )
            v = hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.self_attn.v_proj.weight'].view(
                *new_kv_tensor_shape
            )

            qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])  # .cuda()
            heads_per_group = head_num // num_query_groups
            for count in range(num_query_groups):
                qkv_weights = torch.cat(
                    (qkv_weights, q[count * heads_per_group : (count + 1) * heads_per_group, :, :])
                )
                qkv_weights = torch.cat((qkv_weights, k[count : count + 1, :, :]))
                qkv_weights = torch.cat((qkv_weights, v[count : count + 1, :, :]))
            qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

            param_to_weights = lambda param: param.float()
            new_state_dict[f'model.decoder.layers.{i}.self_attention.linear_qkv.weight'] = param_to_weights(
                qkv_weights
            )

            new_state_dict[f'model.decoder.layers.{i}.self_attention.linear_proj._extra_state'] = (
                nemo_model_from_hf.state_dict()[f'model.decoder.layers.{i}.self_attention.linear_proj._extra_state']
            )
            new_state_dict[f'model.decoder.layers.{i}.self_attention.linear_qkv._extra_state'] = (
                nemo_model_from_hf.state_dict()[f'model.decoder.layers.{i}.self_attention.linear_qkv._extra_state']
            )
        if symb == "-":
            new_state_dict[f'model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.pre_ff_layernorm.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mlp.linear_fc1.weight'] = torch.cat(
                [
                    hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.feed_forward.gate_proj.weight'],
                    hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.feed_forward.up_proj.weight'],
                ]
            )
            new_state_dict[f'model.decoder.layers.{i}.mlp.linear_fc2.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.feed_forward.down_proj.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mlp.linear_fc1._extra_state'] = nemo_model_from_hf.state_dict()[
                f'model.decoder.layers.{i}.mlp.linear_fc1._extra_state'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mlp.linear_fc2._extra_state'] = nemo_model_from_hf.state_dict()[
                f'model.decoder.layers.{i}.mlp.linear_fc2._extra_state'
            ]
        if symb == "O":
            new_state_dict[f'model.decoder.layers.{i}.mlp.input_layernorm.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.pre_ff_layernorm.weight'
            ]
            new_state_dict[f'model.decoder.layers.{i}.mlp.router.weight'] = hf_model.state_dict()[
                f'model.layers.{hf_jamba_layer}.feed_forward.router.weight'
            ]
            for j in range(nemo_config.model.num_moe_experts):
                new_state_dict[f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1.weight'] = (
                    torch.cat(
                        [
                            hf_model.state_dict()[
                                f'model.layers.{hf_jamba_layer}.feed_forward.experts.{j}.gate_proj.weight'
                            ],
                            hf_model.state_dict()[
                                f'model.layers.{hf_jamba_layer}.feed_forward.experts.{j}.up_proj.weight'
                            ],
                        ]
                    )
                )
                new_state_dict[f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc2.weight'] = (
                    hf_model.state_dict()[f'model.layers.{hf_jamba_layer}.feed_forward.experts.{j}.down_proj.weight']
                )
                new_state_dict[
                    f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1._extra_state'
                ] = nemo_model_from_hf.state_dict()[
                    f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1._extra_state'
                ]
                new_state_dict[
                    f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc2._extra_state'
                ] = nemo_model_from_hf.state_dict()[
                    f'model.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc2._extra_state'
                ]

    nemo_model_from_hf.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_hf = nemo_model_from_hf.to(dtype=dtype)

    nemo_model_from_hf.save_to(args.output_path)
    logging.info(f'Jamba NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
