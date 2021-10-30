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
"""
Updates a .nemo file with average weights.
"""

import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.utils import logging


def main():
    folder_path = sys.argv[1]
    checkpoint_paths = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith('.ckpt')]
    """ < Checkpoint Averaging Logic > """
    # load state dicts
    n = len(checkpoint_paths)
    avg_state = None

    logging.info(f"Averaging {n} checkpoints ...")

    for ix, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location='cpu')

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        if ix == 0:
            # Initial state
            avg_state = checkpoint

            logging.info(f"Initialized average state dict with checkpoint : {path}")
        else:
            # Accumulated state
            for k in avg_state:
                avg_state[k] = avg_state[k] + checkpoint[k]

            logging.info(f"Updated average state dict with state from checkpoint : {path}")

    for k in avg_state:
        if str(avg_state[k].dtype).startswith("torch.int"):
            # For int type, not averaged, but only accumulated.
            # e.g. BatchNorm.num_batches_tracked
            pass
        else:
            avg_state[k] = avg_state[k] / n

    # Save model
    ckpt_name = os.path.join(folder_path, 'model_weights.ckpt')
    torch.save(avg_state, ckpt_name)
    logging.info(f"Averaged pytorch checkpoint saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
