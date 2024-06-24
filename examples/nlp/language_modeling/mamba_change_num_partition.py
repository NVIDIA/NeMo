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
import shutil
import tarfile
import tempfile
from argparse import ArgumentParser
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.parts.nlp_overrides import (
    NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState

"""
Usage:

### Tensor Parallelism and Pipeline Parallelism conversion ###

# Megatron Mamba
python /home/ataghibakhsh/NeMo/examples/nlp/language_modeling/mamba_change_num_partition.py \
    --model_file=/home/ataghibakhsh/adlr_mamba2/mamba2-hybrid-8b-3t-4k.nemo \
    --target_file=/home/ataghibakhsh/TP4-ADLR-mamba-hybrid/mamba2-TP4.nemo \
    --tensor_model_parallel_size=1 \
    --target_tensor_model_parallel_size=4 \
    --pipeline_model_parallel_size=1 \
    --target_pipeline_model_parallel_size=1 \
    --precision=bf16

python /home/ataghibakhsh/NeMo/examples/nlp/language_modeling/mamba_change_num_partition.py \
    --model_file=/home/ataghibakhsh/temp_mamba_dir_test/megatron_jamba/2024-06-22_14-25-30/checkpoints/megatron_jamba.nemo \
    --target_file=/home/ataghibakhsh/TEST-TP8/megatron_jamba.nemo \
    --tensor_model_parallel_size=-1 \
    --target_tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=-1 \
    --target_pipeline_model_parallel_size=1 \
    --precision=bf16

### Only Tensor Parallelism conversion ###

To the above commands, add the following argument: `--tp_conversion_only`

# Note: This requires that the pipeline_model_parallel_size and tgt_pipeline_model_parallel_size is set to 1.

### Large Models conversion ###

When converting large models, ** always ** ensure that you pre-extract the nemo model and then only perform conversion

$ mkdir "unpacked_nemo_file"
$ tar -xvf "<path to nemo file>" -C "<absolute path to pwd>/unpacked_nemo_file/"

python megatron_change_num_partitions.py \
    ...
    --model_extracted_dir="<Absolute path to pwd>/unpacked_nemo_file/"

### Model Classes ###

# NOTE: Conversion of other model types.
# Default model type is MegatronGPTModel, if you want another model you need to pass classpath of the model
# For example - MegatronT5Model -

python megatron_change_num_partitions.py \
    ...
    --model_class="nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model"

# Additional arguments:

--num_gpu_per_node: Number of GPUs per node. Default is 8.
--megatron_legacy: Whether the model is a legacy Megatron model or not. Default is False. May be unsuported for
    Pipeline Parallelism change.
--tokenizer_model_path: Path to tokenizer model. Default is None. When not None, overrides the tokenizer model path
    in the model config.
--tokenizer_vocab_file: Path to tokenizer vocab file. Default is None. When not None, overrides the tokenizer vocab
    file in the model config.

# Comments

Passing --tensor_model_parallel_size=-1 or --pipeline_model_parallel_size=-1 will automatically infer the size from the
model config.

"""


import argparse
import copy
import os
import re
import shutil
from collections import OrderedDict

import torch

tp_split_dim = {
    'word_embeddings.weight': 0,
    'norm.weight': -1,
    'final_norm.weight': -1,
    'output_layer.weight': 0,
    # mamba1/2
    'A_log': 0,
    'D': 0,
    'dt_bias': 0,
    'in_proj.weight': 0,
    'conv1d.weight': 0,
    'conv1d.bias': 0,
    'x_proj.weight': 1,
    'dt_proj.weight': 0,
    'dt_proj.bias': 0,
    'out_proj.weight': 1,
    'mixer.norm.weight': 0,
    # mlp
    'linear_fc1.layer_norm_weight': -1,
    'linear_fc1.weight': 0,
    'linear_fc2.weight': 1,
    # attention
    'self_attention.linear_proj.weight': 1,
    'self_attention.linear_qkv.layer_norm_weight': -1,
    'self_attention.linear_qkv.weight': 0,
}


def get_split_dim(tensor_name):
    # norm.weight will match tensor_name of mixer.norm.weight and norm.weight, need to distinguish
    if 'norm.weight' in tensor_name:
        if 'mixer.norm.weight' in tensor_name:
            return tp_split_dim['mixer.norm.weight']
        else:
            return tp_split_dim['norm.weight']

    for key in tp_split_dim.keys():
        if key in tensor_name:
            return tp_split_dim[key]
    raise Exception("Unknown tensor name {}".format(tensor_name))


def split_tensor_for_tp(params, key, dim, tensor):

    tp_size = params.target_tensor_model_parallel_size
    tensor_sliced = []
    if dim == -1:
        tensor_sliced = [tensor for i in range(tp_size)]
    else:
        if 'mixer.in_proj.weight' in key and params.mamba_version == 1:
            x, z = torch.split(tensor, [params.mamba_d_inner, params.mamba_d_inner], dim=dim)
            x_sliced = torch.chunk(x, tp_size, dim=dim)
            z_sliced = torch.chunk(z, tp_size, dim=dim)
            for x, z in zip(x_sliced, z_sliced):
                tensor_sliced.append(torch.cat((x, z), dim=dim))

        elif 'mixer.in_proj.weight' in key and params.mamba_version == 2:
            x, z, B, C, dt = torch.split(
                tensor,
                [
                    params.mamba_d_inner,
                    params.mamba_d_inner,
                    params.mamba2_n_groups * params.mamba_d_state,
                    params.mamba2_n_groups * params.mamba_d_state,
                    params.mamba2_n_heads,
                ],
                dim=dim,
            )
            B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-1]))
            C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-1]))

            B_sliced = torch.chunk(B, tp_size, dim=dim)
            C_sliced = torch.chunk(C, tp_size, dim=dim)
            x_sliced = torch.chunk(x, tp_size, dim=dim)
            z_sliced = torch.chunk(z, tp_size, dim=dim)
            dt_sliced = torch.chunk(dt, tp_size, dim=dim)

            tensor_sliced = []
            for x, z, B, C, dt in zip(x_sliced, z_sliced, B_sliced, C_sliced, dt_sliced):
                tensor_sliced.append(torch.cat((x, z, B.flatten(0, 1), C.flatten(0, 1), dt), dim=dim))

        elif 'mixer.conv1d' in key and params.mamba_version == 2:
            x, B, C = torch.split(
                tensor,
                [
                    params.mamba_d_inner,
                    params.mamba2_n_groups * params.mamba_d_state,
                    params.mamba2_n_groups * params.mamba_d_state,
                ],
                dim=dim,
            )
            if 'weight' in key:
                B = torch.reshape(B, (-1, params.mamba_d_state, B.shape[-2], B.shape[-1]))
                C = torch.reshape(C, (-1, params.mamba_d_state, C.shape[-2], C.shape[-1]))
            elif 'bias' in key:
                B = torch.reshape(B, (-1, params.mamba_d_state))
                C = torch.reshape(C, (-1, params.mamba_d_state))
            else:
                raise Exception("Unknown key")

            B_sliced = torch.chunk(B, tp_size, dim=dim)
            C_sliced = torch.chunk(C, tp_size, dim=dim)
            x_sliced = torch.chunk(x, tp_size, dim=dim)

            tensor_sliced = []
            for x, B, C in zip(x_sliced, B_sliced, C_sliced):
                tensor_sliced.append(torch.cat((x, B.flatten(0, 1), C.flatten(0, 1)), dim=dim))
        elif '_extra_state' in key:
            pass
        else:
            tensor_sliced = torch.chunk(tensor, tp_size, dim=dim)

    return tensor_sliced


#################
### Utilities ###
#################


def force_cpu_model(cfg):
    with open_dict(cfg):
        # temporarily set to cpu
        original_cpu_init = cfg.get('use_cpu_initialization', False)
        if 'megatron_amp_O2' in cfg:
            amp_o2_key = 'megatron_amp_O2'
            original_amp_o2 = cfg.megatron_amp_O2
        elif 'megatron_amp_02' in cfg:
            amp_o2_key = 'megatron_amp_02'
            original_amp_o2 = cfg.megatron_amp_02
        else:
            amp_o2_key, original_amp_o2 = None, None

        # Set new values
        cfg.use_cpu_initialization = True
        if amp_o2_key is not None:
            cfg[amp_o2_key] = False

        # Disable sequence parallelism - Not disabling this gives error when converting the the model to TP=1
        original_sequence_parallel = cfg.get('sequence_parallel', None)
        cfg.sequence_parallel = False

    # Setup restore dict
    restore_dict = {'use_cpu_initialization': original_cpu_init}  # 'megatron_amp_O2': original_amp_o2
    if amp_o2_key is not None:
        restore_dict[amp_o2_key] = original_amp_o2
    if original_sequence_parallel is not None:
        restore_dict['sequence_parallel'] = original_sequence_parallel

    return cfg, restore_dict


def restore_model_config(cfg, original_dict):
    with open_dict(cfg):
        for key, val in original_dict.items():
            logging.info(f"Restoring model config key ({key}) from {cfg[key]} to original value of {val}")
            cfg[key] = val
    return cfg


#################
### Utilities ###
#################


def compute_tp_splits(
    param_name, param, partitions, global_idx, tp_size, pp_size, pp_rank, pp_split_rank, megatron_legacy, model_cfg
):
    """
    Function to compute the splits required for tensor-parallelism.

    Args:
        param_name: Name of the current parameter of the current model (TP X PP Y)
        param: Value of the current parameter of the current compute_tp_splitsmodel (TP X PP Y)
        partitions: Partitions of the flattened parameter of the current model (TP 1 PP 1)
        global_idx: The index used to select the parameter in the global partition.
        tp_size: Int, tensor-parallelism size.
        pp_size: Int, pipeline-parallelism size.
        pp_rank: Int, pipeline-parallelism rank.
        pp_split_rank: Int, pipeline-parallelism split rank. This should be > 1 if TP is being used with EncDec models (T5)
        megatron_legacy: Bool, whether the model is a legacy Megatron model or not.
        model_cfg: The model config as a OmegaConf DictConfig.

    Returns:
        List of torch tensors, each of which is a split of the current parameter.
    """
    # alias the global index to idx
    idx = global_idx

    fast_glu_activation = str(model_cfg.get('activation', '')).lower() in ['fast-geglu', 'fast-swiglu', 'fast-reglu']

    if param.shape == partitions[0][idx].shape:
        split = [partitions[0][idx].data] * tp_size
        logging.debug(">> Perfect match, no splitting needed")
    elif param.shape[0] == partitions[0][idx].shape[0]:
        split = torch.split(partitions[0][idx].data, param.shape[-1], dim=-1)
    else:
        # For T5-converted weights, the splitting needs to be strided such that q,k,v weights are bunched together on each tensor-parallel rank.
        if '.query_key_value.' in param_name and megatron_legacy:  # weight or bias
            split_dim = partitions[0][idx].data.shape[0]
            if split_dim % (tp_size * 3) != 0:
                raise ValueError(
                    f"Can not split Q,K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 3}."
                )
            tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 3, dim=0)
            split = []
            for i in range(tp_size):
                tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 3, tp_size)])
                split.append(tp_qkv)
        elif '.key_value.' in param_name and megatron_legacy:  # weight or bias
            split_dim = partitions[0][idx].data.shape[0]
            if split_dim % (tp_size * 2) != 0:
                raise ValueError(
                    f"Can not split K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 2}."
                )
            tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 2, dim=0)
            split = []
            for i in range(tp_size):
                tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 2, tp_size)])
                split.append(tp_qkv)
        elif ('dense_h_to_4h' in param_name or 'linear_fc1' in param_name) and fast_glu_activation:
            # For Megatron GPT model with Fast Glu activation
            # Handle gated linear units
            # concat all the first halves ('W's) and all the second halves ('V's)
            w_split, k_split = torch.chunk(partitions[0][idx].data, 2, dim=0)
            w_split = torch.chunk(w_split, tp_size, dim=0)
            k_split = torch.chunk(k_split, tp_size, dim=0)
            split = [torch.cat(weights, dim=0) for weights in zip(w_split, k_split)]  # split per tp rank

        # Regular split for Megatron and NeMo-Megatron models.
        else:
            split = torch.split(partitions[0][idx].data, param.shape[0], dim=0)

    return split


def write_tp_pp_split(model, splits, app_state, tp_size, pp_rank, write_path):
    """
    Function to write the given TP PP split to NeMo File.

    Save each of the TP ranks in reverse order
    This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    The final rank will then save a new NeMo file with all other ranks inside.

    Args:
        model: The model corresponding to the current TP PP split. Contains partial parameters.
        splits: Nested List of tensors containing the TP splits of the current model given current PP rank.
            Indexed as splits[idx][tp_rank].
        app_state: AppState object.
        tp_size:  The global tensor-parallel size of the final model.
        pp_rank: The local pipeline parallel rank of the final model.
        write_path: The path to save the NeMo file.
    """
    for tp_rank in range(tp_size - 1, -1, -1):
        app_state.pipeline_model_parallel_rank = pp_rank
        app_state.tensor_model_parallel_rank = tp_rank

        idx = 0
        for name, param in model.named_parameters():
            split_val = splits[idx][tp_rank].clone()

            if param.shape != split_val.shape:
                raise RuntimeError(
                    f"Can not handle parameter {name}, required shape: {param.shape}, split shape: {split_val.shape}."
                )
                # logging.info(
                #     f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, split shape: {split_val.shape}. Padding to match required size."
                # )

                # if split_val.shape[1:] == param.shape[1:]:
                #     pad = [0, 0] * len(split_val.shape)
                #     pad[-1] = param.shape[0] - split_val.shape[0]
                #     split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                # elif split_val.shape[:-1] == param.shape[:-1]:
                #     pad = [0, param.shape[-1] - split_val.shape[-1]]
                #     split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                # else:
                #     raise RuntimeError(
                #         f"Can not handle parameter {name}, required shape: {param.shape}, split shape: {split_val.shape}."
                #     )

            param.data = split_val
            idx += 1

        if write_path is not None:
            logging.info(f"Writing pp rank {pp_rank} tp rank {tp_rank} to file {write_path}")
            model.save_to(write_path)


def debug_log_split_param_diff(idx, param, param_name, partitions):
    # Log some useful comparison of tensors that are being mapped.
    # Note that the global param index for layers and modules may be different but the shapes
    # and semantics of the layer should match.
    logging.debug(f"Index: {idx} Model Params : {param_name} - {param.shape}")
    logging.debug(f"Index: {idx} Global params: {partitions[1][idx]} - {partitions[0][idx].shape}")


##################
### Converters ###
##################


def split_tp_partition_only(args, model, original_model, tp_size, write_path=None, megatron_legacy=False):

    if tp_size < 1:
        raise ValueError("TP size must to be >= 1.")

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = 1
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    app_state.pipeline_model_parallel_rank = 0
    app_state.tensor_model_parallel_rank = tp_size - 1

    idx = 0
    splits = []

    for ii, (key, original_tensor) in enumerate(original_model.model.state_dict().items()):
        try:
            layer_num = int(re.findall(r'\d+', key)[0])
            new_key = key.replace(str(layer_num), str(layer_num), 1)
        except:
            new_key = key

        if '_extra_state' not in new_key:
            split_dim = get_split_dim(new_key)
            split = split_tensor_for_tp(args, new_key, split_dim, original_tensor)

            # split = split_tensor_for_tp(
            #     param_name,
            #     param,
            #     partitions,
            #     idx,
            #     tp_size,
            #     pp_size=1,
            #     pp_rank=0,
            #     pp_split_rank=0,
            #     megatron_legacy=megatron_legacy,
            #     model_cfg=model.cfg,
            # )
            splits.append(split)
            idx += 1

    # Save each of the TP ranks in reverse order
    # This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    # The final rank will then save a new NeMo file with all other ranks inside.
    write_tp_pp_split(model, splits, app_state, tp_size, pp_rank=0, write_path=write_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default=None, required=False, help="Path to source .nemo file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to write target .nemo file")
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=-1, required=False, help="TP size of source model"
    )
    parser.add_argument("--target_tensor_model_parallel_size", type=int, required=True, help="TP size of target model")
    parser.add_argument(
        '--pipeline_model_parallel_size', type=int, default=-1, required=False, help='PP size of source model'
    )
    parser.add_argument(
        '--target_pipeline_model_parallel_size', type=int, required=True, help='PP size of target model'
    )
    parser.add_argument(
        '--target_pipeline_model_parallel_split_rank', type=int, default=0, help='PP rank to split for Enc-Dec models'
    )
    parser.add_argument(
        '--virtual_pipeline_model_parallel_size', type=int, default=None, help='Virtual Pipeline parallelism size'
    )
    parser.add_argument(
        '--ckpt_name', type=str, default=None, help='Checkpoint name to load from for Virtual Parallel'
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="nemo.collections.nlp.models.language_modeling.megatron_jamba_model.MegatronJambaModel",
        help="NeMo model class. This script should support all NeMo megatron models that use Tensor Parallel",
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")
    parser.add_argument('--num_gpu_per_node', default=8, type=int, help='Number of GPUs per node')
    parser.add_argument(
        "--megatron_legacy",
        action="store_true",
        help="Converter for legacy megatron modles that have different q,k,v weight splits",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer model path if your model uses a tokenizer model as an artifact. This is needed if your model uses a sentencepiece tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_vocab_file",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer model path if your model uses a tokenizer model as an artifact. This is needed if your model uses a sentencepiece tokenizer.",
    )
    parser.add_argument('--hparams_file', type=str, default=None, help='Path to hparams file from PTL training')
    parser.add_argument(
        '--tp_conversion_only', default=True, action='store_true', help='Only convert TP model to TP model'
    )
    parser.add_argument('--model_extracted_dir', type=str, default=None, help='Path to pre-extracted model directory')

    parser.add_argument('--d-model', type=int, default=4096)
    parser.add_argument('--mamba-version', type=int, default=2)
    parser.add_argument('--mamba-d-state', type=int, default=128)
    parser.add_argument('--mamba2-n-groups', type=int, default=8)
    parser.add_argument('--mamba2-head-dim', type=int, default=64)

    args = parser.parse_args()

    args.mamba_d_inner = args.d_model * 2
    args.mamba2_n_heads = args.mamba_d_inner // args.mamba2_head_dim

    precision = args.precision
    num_gpu_per_node = int(args.num_gpu_per_node)
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))

    if precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            pass
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = precision[2:]

    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback

    # Built target directory if it does not exist
    target_dir = os.path.split(args.target_file)[0]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    tp_size = args.tensor_model_parallel_size
    tgt_tp_size = args.target_tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    tgt_pp_size = args.target_pipeline_model_parallel_size
    pipeline_model_parallel_split_rank = args.target_pipeline_model_parallel_split_rank
    vp_size = args.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    convert_vp = vp_size > 1
    if convert_vp:
        from megatron.core import parallel_state

        parallel_state.set_virtual_pipeline_model_parallel_world_size(vp_size)

        hparams_filepath = args.hparams_file
        if hparams_filepath is None:
            logging.warning(
                '\n\n\n!!!!!!!!!\n'
                'You are converting a model with virtual pipeline parallelism enabled, \n'
                'but have not passed `hparams_file` argument. \n'
                'This will cause each ckpt file to be temporarily laoded onto GPU memory!\n\n'
                'It is highly recommended to pass `hparams_file` argument to avoid this.\n'
            )
    else:
        hparams_filepath = None

    # Import the class of the model
    cls = model_utils.import_class_by_path(args.model_class)

    if args.model_file is None and args.model_extracted_dir is None:
        raise ValueError("Cannot pass model_file and model_extracted_dir as None at the same time.")

    tmp_cfg = cls.restore_from(
        restore_path=args.model_file,
        trainer=Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu", precision=precision),
        map_location=torch.device("cpu"),
        return_config=True,
    )
    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=tmp_cfg.get('native_amp_init_scale', 2**32),
                growth_interval=tmp_cfg.get('native_amp_growth_interval', 1000),
                hysteresis=tmp_cfg.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if tmp_cfg.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        # Set precision None after precision plugins are created as PTL >= 2.1 does not allow both
        # precision plugins and precision to exist
        precision = None
    trainer = Trainer(plugins=plugins, devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")

    if tp_size < 0 or pp_size < 0:
        logging.info(f"Loading model config from {args.model_file} to get TP and PP size")
        model_config_internal = cls.restore_from(
            restore_path=args.model_file,
            trainer=trainer,
            map_location=torch.device("cpu"),
            return_config=True,
        )

        tp_size = model_config_internal.get('tensor_model_parallel_size', 1)
        pp_size = model_config_internal.get('pipeline_model_parallel_size', 1)

    # Check if TP conversion only
    tp_conversion_only = args.tp_conversion_only
    if tp_conversion_only:
        logging.info("Converting TP model to TP model only")

        if pp_size > 1:
            raise ValueError("Provided `--tp_conversion_only` but `--pipeline_model_parallel_size` > 1")

        if tgt_pp_size > 1:
            raise ValueError("Provided `--tp_conversion_only` but `--target_pipeline_model_parallel_size` > 1")

        if pipeline_model_parallel_split_rank > 0:
            raise ValueError("Provided `--tp_conversion_only` but `--target_pipeline_model_parallel_split_rank` > 0")

        # Force PP size to 1
        pp_size = 1
        tgt_pp_size = 1
        pipeline_model_parallel_split_rank = 0

    if vp_size is None or vp_size < 0:
        vp_size = 1

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = pp_size
    app_state.tensor_model_parallel_size = tp_size

    if vp_size > 1:
        app_state.virtual_pipeline_model_parallel_size = vp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    world_size = pp_size * tp_size  # pseudo world size for simulating load of a specific rank on a single gpu

    app_state.tensor_model_parallel_rank = 0
    app_state.pipeline_model_parallel_rank = 0

    # Extract tokenizer artifact from the model to temp directory
    logging.info("Extracting tokenizer artifact from NeMo file...")
    temp_dir = tempfile.mkdtemp()
    tokenizer_model_path = None
    with tarfile.open(args.model_file, "r") as tar:
        for member in tar.getmembers():
            if '.model' in member.name:
                extracted_file = tar.extractfile(member)
                extracted_file_path = os.path.join(temp_dir, member.name)

                if tokenizer_model_path is None:
                    logging.info(f"Found tokenizer. Extracting {member.name} to {extracted_file_path}")

                    tokenizer_model_path = extracted_file_path
                    with open(extracted_file_path, "wb") as f:
                        f.write(extracted_file.read())
                else:
                    if args.tokenizer_model_path is None:
                        logging.warning(
                            f"\n\nFound multiple tokenizer artifacts in the model file.\n"
                            f"Using only {tokenizer_model_path}.\n"
                            f"If this is incorrect, manually pass the correct tokenizer using "
                            f"`--tokenizer_model_path`.\n\n"
                        )

    # If input model has TP > 1 or PP > 1
    # Reconstruct the model to have TP = 1 and PP = 1
    # Note that this is a forward loop that will process PP [0..N] TP [0..M] in sequential order.

    # If input model has TP = 1 and PP = 1
    app_state.model_parallel_size = 1

    save_restore_connector = NLPSaveRestoreConnector()

    if args.model_extracted_dir is not None:
        logging.info(f"Using extracted model directory: {args.model_extracted_dir}")
        save_restore_connector.model_extracted_dir = args.model_extracted_dir

    if args.model_file is not None:
        model_filepath = args.model_file
    else:
        model_filepath = args.model_extracted_dir

    tmp_cfg = cls.restore_from(
        restore_path=model_filepath,
        trainer=trainer,
        map_location=torch.device("cpu"),
        save_restore_connector=save_restore_connector,
        return_config=True,
    )

    tmp_cfg, restore_dict = force_cpu_model(tmp_cfg)

    model = cls.restore_from(
        restore_path=model_filepath,
        trainer=trainer,
        map_location=torch.device("cpu"),
        save_restore_connector=save_restore_connector,
        override_config_path=tmp_cfg,
    )

    original_model = cls.restore_from(
        restore_path=model_filepath,
        trainer=trainer,
        map_location=torch.device("cpu"),
        save_restore_connector=save_restore_connector,
        override_config_path=tmp_cfg,
    )
    original_model = original_model.to('cpu')
    original_model._save_restore_connector = NLPSaveRestoreConnector()
    original_model.freeze()
    original_model.to(dtype=dtype)

    # for k, v in model.model[0].state_dict().items():
    #     if v is not None:
    #         print(f"k = {k}, v = {v.shape}")
    # import sys
    # sys.exit()
    model.to(dtype=dtype)

    restore_model_config(model.cfg, restore_dict)

    # If target model has TP > 1 or PP > 1
    if tgt_pp_size > 1 or tgt_tp_size > 1:

        # Preserve the TP 1 PP 1 model parameters and names
        global_params = []
        global_params.append([p for n, p in model.named_parameters()])  # params
        global_params.append([n for n, p in model.named_parameters()])  # names

        logging.debug("Global parameters:")
        for idx, (name, p) in enumerate(zip(global_params[1], global_params[0])):
            logging.debug(f"{name} - {p.shape}")

        logging.info(f"TP 1 PP 1 Number of Parameters : {len(global_params[0])}")

        world_size = (
            tgt_pp_size * tgt_tp_size
        )  # pseudo world size for simulating load of a specific rank on a single gpu
        new_global_batch_size = model.cfg.micro_batch_size * world_size
        old_global_batch_size = model.cfg.get('global_batch_size', model.cfg.micro_batch_size)

        global_offset = len(global_params[0]) - 1  # -1 cause this indexes the array, range [0, L-1]
        logging.info(f"Final layer offset for parameters: {global_offset}")

        for pp_rank in range(tgt_pp_size - 1, -1, -1):  # reverse order

            with open_dict(model.cfg):
                model.cfg.pipeline_model_parallel_size = tgt_pp_size
                model.cfg.tensor_model_parallel_size = tgt_tp_size

                if 'pipeline_model_parallel_split_rank' in model.cfg:
                    if pipeline_model_parallel_split_rank > 0:
                        model.cfg.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank
                    elif pp_size > 1:
                        logging.warning(
                            f"Model config has `pipeline_model_parallel_split_rank` set to "
                            f"{model.cfg.pipeline_model_parallel_split_rank} and target PP "
                            f"size is {tgt_pp_size}. "
                            f"Provided `pipeline_model_parallel_split_rank` is "
                            f"{pipeline_model_parallel_split_rank}. "
                            f"Be careful that the model config is correct "
                            f"if encoder-decoder models are being converted."
                        )

                model.cfg.global_batch_size = old_global_batch_size  # Used for restoration

            # Override flag that forces Model to use AppState instead of Trainer
            # to determine the world size, global and local rank
            # Used for simulating load of a specific rank on a single gpu
            os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

            # Compute the global rank
            global_rank = (
                pp_rank * tgt_tp_size + 0
            )  # tp_rank = 0 needed just for modules, all TP will be merged to this PP rank

            # Update AppState
            app_state.world_size = world_size
            app_state.global_rank = global_rank
            app_state.local_rank = global_rank % num_gpu_per_node
            app_state.pipeline_model_parallel_size = tgt_pp_size
            app_state.tensor_model_parallel_size = tgt_tp_size
            app_state.model_parallel_size = (
                app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
            )

            trainer = Trainer(plugins=plugins, devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")
            if args.tokenizer_model_path is not None:
                with open_dict(model.cfg):
                    model.cfg.tokenizer.model = args.tokenizer_model_path

            else:
                if tokenizer_model_path is None:
                    logging.warning("Could not extract tokenizer model file from checkpoint.")

                else:
                    # Extract tokenizer info
                    with open_dict(model.cfg):
                        model.cfg.tokenizer.model = tokenizer_model_path

            model.cfg, restore_dict = force_cpu_model(model.cfg)

            model = cls(model.cfg, trainer)
            model = model.to('cpu')
            model._save_restore_connector = NLPSaveRestoreConnector()
            model.freeze()
            model.to(dtype=dtype)
            # for k, v in model.model.state_dict().items():
            #     if v is not None:
            #         print(f"k = {k}, v = {v.shape}")
            # import sys
            # sys.exit()
            restore_model_config(model.cfg, restore_dict)

            # Update global batch size
            if old_global_batch_size % new_global_batch_size != 0 or old_global_batch_size < new_global_batch_size:
                logging.info(
                    f"Global batch size {old_global_batch_size} is not divisible by new global batch size {new_global_batch_size}."
                    f" The model config will be updated with new global batch size {new_global_batch_size}."
                )
                with open_dict(model.cfg):
                    model.cfg.global_batch_size = new_global_batch_size

            logging.info(f"Global rank: {global_rank} Local rank: {app_state.local_rank} World size: {world_size}")
            logging.info(f"PP rank: {pp_rank} TP rank: {0}")
            logging.info(f"TP 1 PP 1 Number of Layers : {len(global_params[0])}")
            logging.info(f"Remaining layer offset for parameters: {global_offset}")
            logging.info("\n")

            # Special case for TP conversion only mode
            if tp_conversion_only:
                logging.info(f"Skipping PP split due to flag `--tp_conversion_only`")
                split_tp_partition_only(
                    args, model, original_model, tgt_tp_size, args.target_file, args.megatron_legacy
                )
                break


if __name__ == '__main__':
    main()
