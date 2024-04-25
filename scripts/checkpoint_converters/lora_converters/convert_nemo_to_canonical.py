#!/usr/bin/env
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
Merge lora weights into a base GPT LM.
Supports any TP and PP the LoRA model is trained on, and no need to specify TP/PP when running this script

Example usage:
python scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=gpu \   (use 'cpu' if model cannot fit in memory)
    gpt_model_file=<path to base model nemo file or extracted folder> \
    lora_model_path=<path to megatron_gpt_peft_lora_tuning.nemo> \
    merged_model_path=<output nemo file>

"""
import os
import pdb
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf, open_dict
from scripts.nlp_language_modeling.merge_lora_weights.merge import replace_number_add_offset

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def rename_keys(key):
    new_keys = []
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.q_adapter."))
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.k_adapter."))
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.v_adapter."))
    return new_keys


def convert(lora_weights, lora_model_cfg):
    assert len(lora_weights) == 1, "Only single TP supported for now"
    if (
        lora_model_cfg.get("num_query_groups", lora_model_cfg.num_attention_heads)
        != lora_model_cfg.num_attention_heads
    ):
        kv_channels = int(lora_model_cfg.hidden_size / lora_model_cfg.num_attention_heads)
        kv_size = int(lora_model_cfg.num_query_groups * kv_channels)
    else:
        kv_size = int(lora_model_cfg.hidden_size)
    q_size = lora_model_cfg.hidden_size
    k_size, v_size = kv_size, kv_size

    keys_to_update = []
    for key in lora_weights[0].keys():
        if "lora_kqv_adapter" in key:
            keys_to_update.append(key)

    for key in keys_to_update:
        if "linear_in" in key:
            for new_key in rename_keys(key):
                lora_weights[0][new_key] = lora_weights[0][key]
        elif "linear_out" in key:
            srt = 0
            for new_key, size in zip(rename_keys(key), [q_size, k_size, v_size]):
                print(size, new_key)
                lora_weights[0][new_key] = lora_weights[0][key][srt : srt + size]
                srt = srt + size

        lora_weights[0].pop(key)
    return lora_weights


def convert_lora(lora_nemo, save_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        config_file = f"{tmpdir}/model_config.yaml"
        lora_config = OmegaConf.load(config_file)
        tp_size = lora_config.tensor_model_parallel_size
        pp_size = lora_config.pipeline_model_parallel_size

        lora_state_dict = [{}] * tp_size

        for pp in range(pp_size):
            for tp in range(tp_size):
                if tp_size == 1:
                    ckpt_file = f"{tmpdir}/model_weights.ckpt"
                elif pp_size == 1:
                    ckpt_file = f"{tmpdir}/mp_rank_{tp:02d}/model_weights.ckpt"
                else:
                    ckpt_file = f"{tmpdir}/tp_rank_{tp:02d}_pp_rank_{pp:03d}/model_weights.ckpt"

                l = torch.load(ckpt_file, map_location=torch.device('cpu'))
                if pp == 0:
                    lora_state_dict[tp] = l
                else:
                    # calculate layer offset
                    layer_offset = lora_config.num_layers // pp_size * pp
                    for key, value in l.items():
                        new_key = replace_number_add_offset(key, layer_offset)
                        lora_state_dict[tp][new_key] = value

        with open_dict(lora_config):
            lora_config.peft.lora_tuning.variant = "canonical"
        with open(f"{tmpdir}/model_config.yaml", "w") as f:
            OmegaConf.save(lora_config, f)
        lora_state_dict = convert(lora_state_dict, lora_config)
        torch.save(lora_state_dict[0], f"{tmpdir}/model_weights.ckpt")  # TODO: currently suport tp=1
        NLPSaveRestoreConnector._make_nemo_file_from_folder(save_path, tmpdir)

    return lora_state_dict, lora_config


def fix_for_O2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model.module." not in k:
            new_state_dict[k.replace('model.', 'model.module.')] = v
    return new_state_dict


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo style (fused) lora checkpoint in .nemo file format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the canonical (unfused) lora .nemo file.",
    )
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    convert_lora(args.lora_path, args.output_path)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
