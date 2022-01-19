# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
Conversion script to convert PTL checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> megatron_gpt_ckpt_to_nemo.py.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --nemo_file_path <path_to_output_nemo_file> \
     --tensor_model_parallel_size <tensor_model_parallel_size>
"""

import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=True,
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=6.34-step=649-last.ckpt",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument("--model_type", type=str, required=True, default="gpt", choices=["gpt", "t5", "bert"])

    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))

    args = parser.parse_args()
    return args


def convert(rank, world_size, args):

    app_state = AppState()
    app_state.data_parallel_rank = 0
    trainer = Trainer(gpus=args.tensor_model_parallel_size)
    # TODO: reach out to PTL For an API-safe local rank override
    trainer.accelerator.training_type_plugin._local_rank = rank

    if args.tensor_model_parallel_size is not None and args.tensor_model_parallel_size > 1:
        # inject model parallel rank
        checkpoint_path = os.path.join(args.checkpoint_folder, f'mp_rank_{rank:02d}', args.checkpoint_name)
    else:
        checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_name)

    if args.model_type == 'gpt':
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'bert':
        model = MegatronBertModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    elif args.model_type == 't5':
        model = MegatronT5Model.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()
    world_size = args.tensor_model_parallel_size

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size
        )

    convert(args.local_rank, world_size, args)
