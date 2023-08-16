# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
Example: python scripts/checkpoint_averaging/average_model_checkpoints.py +name=model +checkpoint_dir=/lustre/fsw/swdl/swdl-langspeech/igitman/llm/sft-results/gpt_43b_sft_gsm8k_coding_chat-nodef-ans/checkpoints/
"""

import os

import torch
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


def process_config(cfg: OmegaConf, i):
    checkpoint_dir = os.path.join(cfg['checkpoint_dir'], f"mp_rank_0{i}")

    checkpoint_paths = [
        os.path.join(checkpoint_dir, x)
        for x in os.listdir(checkpoint_dir)
        if x.endswith('.ckpt') and not x.endswith('-last.ckpt')
    ]
    return cfg.name, checkpoint_paths, checkpoint_dir


@hydra_runner(config_path=None, config_name=None)
def main(cfg):
    # repeating for 8 ranks
    for i in range(8):
        print(f"Processing mp_rank_0{i}")
        name_prefix, checkpoint_paths, checkpoint_dir = process_config(cfg, i)

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
        ckpt_name = os.path.join(checkpoint_dir, name_prefix + '-averaged.ckpt')
        torch.save({'state_dict': avg_state}, ckpt_name)

        logging.info(f"Averaged pytorch checkpoint saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
