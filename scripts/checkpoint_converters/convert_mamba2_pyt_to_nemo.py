import json
import os
from argparse import ArgumentParser

import torch
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_jamba_model import MegatronJambaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

'''
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path /home/ataghibakhsh/mamba2_ckpt/mamba2-130m --output_path /home/ataghibakhsh/forks/mamba_130m.nemo
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path /home/ataghibakhsh/gitlab/rogers/mamba_share --output_path /home/ataghibakhsh/mmm_mamba2.nemo
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path adlr_mamba2/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt --output_path /home/ataghibakhsh/adlr_mamba2/mamba2-8b-3t-4k.nemo
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path adlr_mamba2/mamba2-hybrid-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt --output_path /home/ataghibakhsh/adlr_mamba2/mamba2-hybrid-8b-3t-4k.nemo
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path adlr_mamba2/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt --output_path /home/ataghibakhsh/adlr_mamba2/mamba2-hybrid-random.nemo
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path adlr_mamba2/gpt3-8b-multi-3.5t-base/release/mp_rank_00/model_optim_rng.pt --output_path /home/ataghibakhsh/adlr_mamba2/gpt3-base.nemo
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
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    args = parser.parse_args()
    return args


def convert_pyt(args):

    mmm = False

    with open(args.input_name_or_path + '/config.json', 'r') as config_file:
        pytorch_config = json.load(config_file)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.hidden_size = pytorch_config['d_model']
    nemo_config.model.num_layers = pytorch_config['n_layer']
    if mmm:
        nemo_config.model.vocab_size = pytorch_config['vocab_size']
    else:
        nemo_config.model.vocab_size = pytorch_config['vocab_size'] + 11
    nemo_config.model.make_vocab_size_divisible_by = pytorch_config['pad_vocab_size_multiple']
    nemo_config.model.hybrid_override_pattern = "M" * nemo_config.model.num_layers

    nemo_config.model.use_cpu_initialization = True

    logging.info(f"Loading Mamba2 Pytorch checkpoint : `{args.input_name_or_path}`")

    pytorch_model_weights = torch.load(args.input_name_or_path + '/pytorch_model.bin', map_location='cpu')

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()
    nemo_model_from_pyt = MegatronJambaModel(nemo_config.model, trainer)

    from src.models.config_mamba import MambaConfig
    from src.models.mixer_seq_simple import Mamba2LMHeadModel, MixerModel

    mamba_cfg = MambaConfig(
        d_model=pytorch_config['d_model'],
        n_layer=pytorch_config['n_layer'],
        d_intermediate=0,  # pytorch_config['d_intermediate'],
        vocab_size=pytorch_config['vocab_size'],
        ssm_cfg=None,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=False,
    )

    pytorch_model = Mamba2LMHeadModel(
        config=mamba_cfg,
        device=None,
        dtype=None,
    )

    new_state_dict = {}

    new_state_dict['model.embedding.word_embeddings.weight'] = pytorch_model_weights['backbone.embedding.weight']
    new_state_dict['model.decoder.final_norm.weight'] = pytorch_model_weights['backbone.norm_f.weight']
    new_state_dict['model.output_layer.weight'] = pytorch_model_weights['lm_head.weight']
    for i in range(nemo_config.model.num_layers):

        new_state_dict[f'model.decoder.layers.{i}.mixer.A_log'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.A_log'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.D'] = pytorch_model_weights[f'backbone.layers.{i}.mixer.D']
        new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.weight'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.conv1d.weight'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.bias'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.conv1d.bias'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.in_proj.weight'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.in_proj.weight'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.dt_bias'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.dt_bias'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.out_proj.weight'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.out_proj.weight'
        ]
        new_state_dict[f'model.decoder.layers.{i}.mixer.norm.weight'] = pytorch_model_weights[
            f'backbone.layers.{i}.mixer.norm.weight'
        ]
        new_state_dict[f'model.decoder.layers.{i}.norm.weight'] = pytorch_model_weights[
            f'backbone.layers.{i}.norm.weight'
        ]

    pytorch_model.cuda()
    nemo_model_from_pyt.cuda()

    pytorch_model.load_state_dict(dict(pytorch_model_weights), strict=True)
    nemo_model_from_pyt.load_state_dict(new_state_dict, strict=True)

    # print((nemo_model_from_pyt.state_dict()['model.embedding.word_embeddings.weight'] - pytorch_model_weights['tok_embeddings.weight'].cuda()).sum())
    # print((nemo_model_from_pyt.state_dict()['model.decoder.final_norm.weight'] - pytorch_model_weights['norm.weight'].cuda()).sum())
    # print((nemo_model_from_pyt.state_dict()['model.output_layer.weight'] - pytorch_model_weights['output.weight'].cuda()).sum())
    # for i in range(nemo_config.model.num_layers):
    #     print(f'layer {i}')
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.A_log'] - pytorch_model_weights[f'layers.{i}.mixer.A_log'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.D'] - pytorch_model_weights[f'layers.{i}.mixer.D'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.conv1d.weight'] - pytorch_model_weights[f'layers.{i}.mixer.conv1d.weight'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.conv1d.bias'] - pytorch_model_weights[f'layers.{i}.mixer.conv1d.bias'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.in_proj.weight'] - pytorch_model_weights[f'layers.{i}.mixer.in_proj.weight'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.dt_bias'] - pytorch_model_weights[f'layers.{i}.mixer.dt_bias'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.out_proj.weight'] - pytorch_model_weights[f'layers.{i}.mixer.out_proj.weight'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.mixer.norm.weight']- pytorch_model_weights[f'layers.{i}.mixer.norm.weight'].cuda()).sum())
    #     print((nemo_model_from_pyt.state_dict()[f'model.decoder.layers.{i}.norm.weight'] - pytorch_model_weights[f'layers.{i}.norm.weight'].cuda()).sum())

    inpt = torch.randint(10, (1, 10)).cuda()
    out_pyt = pytorch_model.forward(inpt)
    out_nemo = nemo_model_from_pyt.forward(inpt)
    print(f"out_pyt = {out_pyt}")
    print(f"out_nemo = {out_nemo}")

    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_pyt = nemo_model_from_pyt.to(dtype=dtype)
    nemo_model_from_pyt.save_to(args.output_path)
    logging.info(f'Mamba2 NeMo model saved to: {args.output_path}')


def convert_mlm(args):

    a = torch.load(args.input_name_or_path)
    pytorch_model_weights = a['model']
    args_tc = a['args']

    # hybrid_override_pattern = "M" * args_tc.num_layers
    hybrid_override_pattern = 'M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-'

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.hidden_size = args_tc.hidden_size
    nemo_config.model.num_layers = args_tc.num_layers
    nemo_config.model.ffn_hidden_size = args_tc.ffn_hidden_size
    nemo_config.model.vocab_size = args_tc.padded_vocab_size
    nemo_config.model.num_attention_heads = args_tc.num_attention_heads
    nemo_config.model.hybrid_override_pattern = hybrid_override_pattern
    nemo_config.model.num_query_groups = args_tc.num_query_groups
    nemo_config.model.gated_linear_unit = False
    nemo_config.model.layernorm_epsilon = 1e-5

    nemo_config.model.use_cpu_initialization = True
    from megatron.core.transformer.transformer_config import TransformerConfig

    transformer_config = TransformerConfig(
        num_layers=1,  # args_tc.num_layers,
        hidden_size=1,  # args_tc.hidden_size,
        num_attention_heads=1,  # args_tc.num_attention_heads,
        ffn_hidden_size=1,  # args_tc.ffn_hidden_size,
        use_cpu_initialization=False,
        #    rotary_percent=args_tc.rotary_percent,
        normalization="RMSNorm",
        add_bias_linear=args_tc.add_bias_linear,
        gated_linear_unit=False,
        hidden_dropout=args_tc.hidden_dropout,
        num_moe_experts=16,
        moe_aux_loss_coeff=0.001,
        num_query_groups=args_tc.num_query_groups,
        attention_dropout=args_tc.attention_dropout,
    )

    def update_dataclass_from_args(dataclass_obj, args):
        for field in dataclass_obj.__dataclass_fields__:
            if hasattr(args, field):
                setattr(dataclass_obj, field, getattr(args, field))

    update_dataclass_from_args(transformer_config, args_tc)
    from dataclasses import dataclass, fields

    def compare_dataclasses(dc1, dc2):
        differences = {}
        for field in fields(dc1):
            value1 = getattr(dc1, field.name)
            value2 = getattr(dc2, field.name)
            if value1 != value2:
                differences[field.name] = (value1, value2)
        return differences

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()
    nemo_model_from_pyt = MegatronJambaModel(nemo_config.model, trainer)

    # Compare the dataclass instances and print the differences
    differences = compare_dataclasses(transformer_config, nemo_model_from_pyt.transformer_config)

    print("Differences between dataclass instances:")
    for field, (value1, value2) in differences.items():
        print(f"{field}: DataClass1 -> {value1}, DataClass2 -> {value2}")
    # import sys
    # sys.exit()
    new_state_dict = {"model." + key: value for key, value in pytorch_model_weights.items()}

    # pytorch_model.cuda()
    # nemo_model_from_pyt#.cuda()

    # pytorch_model.load_state_dict(dict(pytorch_model_weights), strict=True)
    nemo_model_from_pyt.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_pyt = nemo_model_from_pyt.to(dtype=dtype)
    print(f"nemo_model_from_pyt.model.max_sequence_length = {nemo_model_from_pyt.model.max_sequence_length}")
    data = list(range(128))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
    attention_mask = None
    # out_pyt = pytorch_model.forward(inpt)
    out_nemo = nemo_model_from_pyt.forward(input_ids, position_ids=position_ids, attention_mask=attention_mask)
    # print(f"out_pyt = {out_pyt}")
    print(f"out_nemo = {out_nemo[:, :, 5000:5005]}")

    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_pyt = nemo_model_from_pyt.to(dtype=dtype)

    from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

    nemo_model_from_pyt._save_restore_connector = NLPSaveRestoreConnector()
    nemo_model_from_pyt.cfg.use_cpu_initialization = False
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    nemo_model_from_pyt.save_to(args.output_path)
    logging.info(f'Mamba2 NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    # convert_pyt(args)
    convert_mlm(args)
