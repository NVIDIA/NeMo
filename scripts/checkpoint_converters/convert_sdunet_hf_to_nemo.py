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

r"""
Conversion script to convert HuggingFace Starcoder2 checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_hf_starcoder2_to_nemo.py \
     --input_name_or_path <path_to_sc2_checkpoints_folder> \
     --output_path <path_to_output_nemo_file>
"""


import json
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface UNet checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument("--debug", type=bool, action='store_true', help="Useful for debugging purposes.")
    
    args = parser.parse_args()
    return args

def make_tiny_config(config):
    ''' dial down the config file to make things tractable '''
    return config

def load_unet_ckpt(in_dir):
    # takes a directory as input
    params_file = os.path.join(in_dir, 'config.json')
    assert os.path.exists(params_file)
    with open(params_file, 'r') as fp:
        model_args = json.load(fp)
    if args.debug:
        model_args = make_tiny_config(model_args)

    model = AutoModel.from_pretrained(in_dir)
    ckpt = model.state_dict()
    return model_args, ckpt 

def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")
    args, ckpt = load_unet_ckpt(args.input_name_or_path)
    

if __name__ == '__main__':
    args = get_args()
    convert(args)