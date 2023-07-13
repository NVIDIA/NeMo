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
Usage example:
  python /opt/NeMo/examples/multimodal/generative/stable_diffusion/convert_hf_ckpt_to_nemo.py
    --ckpt_path=path/to/hf.ckpt
    --hparams_file=path/to/saved.yaml
    --nemo_file_path=hf2sd.nemo

Additionally, provide a NeMo hparams file with the correct model architecture arguments. Refer to examples/multimodal/foundation/clip/conf/megatron_clip_config.yaml.
"""

import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cloud_io import load as pl_load

from nemo.collections.multimodal.models.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="Path to checkpoint.")

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=False, default=1)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")
    parser.add_argument("--model_type", type=str, required=False, default="stable_diffusion")

    args = parser.parse_args()
    return args


def mapping_hf_state_dict(hf_state_dict, model):
    nemo_state = model.state_dict()
    new_state_dict = {}
    for k, v in hf_state_dict.items():
        k = 'model.' + k
        # This is not necessary when you turn off model.inductor in config file
        # if 'diffusion_model' in k:
        #     k = k.replace('diffusion_model', 'diffusion_model._orig_mod')
        if 'in_layers' in k or 'out_layers' in k:
            s = k.split('.')
            idx = int(s[-2])
            if idx != 0:
                k = ".".join(s[:-2] + [str(int(idx - 1))] + [s[-1]])
        if k in nemo_state:
            new_state_dict[k] = v

    return new_state_dict


def convert(local_rank, rank, world_size, args):
    app_state = AppState()
    app_state.data_parallel_rank = 0
    num_nodes = world_size // args.gpus_per_node
    if args.bcp:
        trainer = Trainer(
            devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu', plugins=[TorchElasticEnvironment()]
        )
    else:
        trainer = Trainer(devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu')

    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

    # no use atm, use to split ranks in encoder/decoder models.
    if args.pipeline_model_parallel_size > 1 and args.model_type in []:
        if args.pipeline_model_parallel_split_rank is not None:
            app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_split_rank
        else:
            if args.pipeline_model_parallel_size % 2 != 0:
                raise ValueError(
                    f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be even if split rank is not specified."
                )
            else:
                # If split rank is not set, then we set it to be pipeline_model_parallel_size // 2 - this is because in most cases we have the same number of enc/dec layers.
                app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_size // 2
    else:
        app_state.pipeline_model_parallel_split_rank = None

    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()

    checkpoint = pl_load(args.ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    cfg = OmegaConf.load(args.hparams_file)
    if args.model_type == 'stable_diffusion':
        model = MegatronLatentDiffusion(cfg.model, trainer)
    elif args.model_type == 'controlnet':
        model = MegatronControlNet(cfg.model, trainer)

    state_dict = mapping_hf_state_dict(checkpoint, model)

    model.load_state_dict(state_dict)

    model._save_restore_connector = NLPSaveRestoreConnector()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()
    local_rank, rank, world_size = initialize_distributed(args)
    convert(local_rank, rank, world_size, args)
