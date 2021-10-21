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


import os
from argparse import ArgumentParser

import torch.multiprocessing as mp
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
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

    args = parser.parse_args()
    return args


def convert(rank, world_size, args):

    app_state = AppState()
    app_state.data_parallel_rank = 0
    trainer = Trainer(gpus=args.tensor_model_parallel_size)
    # TODO: reach out to PTL For an API-safe local rank override
    trainer.accelerator.training_type_plugin._local_rank = rank
    checkpoint_path = os.path.join(args.checkpoint_folder, f'mp_rank_{rank:02d}', args.checkpoint_name)
    model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    model._save_restore_connector = NLPSaveRestoreConnector()
    model.save_to(args.nemo_file_path)
    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


def main() -> None:
    args = get_args()
    world_size = args.tensor_model_parallel_size
    mp.spawn(convert, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
