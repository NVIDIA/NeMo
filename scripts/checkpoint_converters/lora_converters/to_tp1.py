#!/usr/bin/env
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# taken from https://gitlab-master.nvidia.com/dl/ai-services/nemollm-api/-/blob/main/images/bignlp-training/scripts/lora_to_tp1.py?ref_type=heads

import argparse
import shutil
from tarfile import TarFile
from typing import Dict
import torch


def extract_lora_state_dict_from_tar(tar: TarFile, tp: int) -> Dict[int, torch.Tensor]:
    lora_state_dict = {}

    for i in range(tp):
        ckpt_file = tar.extractfile(f"./mp_rank_0{i}/model_weights.ckpt")
        loaded_state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
        lora_state_dict[i] = loaded_state_dict
    return lora_state_dict


def load_lora(lora_checkpoint_dir: str, tp: int) -> Dict[int, torch.Tensor]:
    lora_state_dict = {}

    for i in range(tp):
        ckpt_file = f"{lora_checkpoint_dir}/mp_rank_0{i}/model_weights.ckpt"
        loaded_state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
        lora_state_dict[i] = loaded_state_dict
    return lora_state_dict


def to_tp1(lora_state_dict):
    tp = len(lora_state_dict)
    target_state_dict = {}
    for key in lora_state_dict[0].keys():
        wt_lora = torch.cat([lora_state_dict[rank][key] for rank in range(tp)], dim=0)
        target_state_dict[key] = wt_lora
    return target_state_dict


def save_tp1(state_dict_tp1, target_lora_checkpoint_dir, lora_checkpoint_dir):
    torch.save(state_dict_tp1, f'{target_lora_checkpoint_dir}/model_weights.ckpt')
    shutil.copy(f"{lora_checkpoint_dir}/model_config.yaml", f"{target_lora_checkpoint_dir}/model_config.yaml")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the directory containing unpacked lora checkpoints.",
    )
    parser.add_argument(
        "--target_lora_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the output directory containing unpacked lora checkpoints.",
    )
    parser.add_argument(
        "--tp", type=int, required=True, help="Tensor parallelism for the input lora checkpoint",
    )
    args = parser.parse_args()

    state_dict = load_lora(lora_checkpoint_dir=args.lora_checkpoint_dir, tp=args.tp)
    state_dict_tp1 = to_tp1(state_dict)
    save_tp1(state_dict_tp1, args.target_lora_checkpoint_dir, args.lora_checkpoint_dir)


if __name__ == "__main__":
    main()
