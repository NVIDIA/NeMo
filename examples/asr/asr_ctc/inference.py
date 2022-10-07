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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts in order to prepare the tokenizer.

```sh

```
"""

import json
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

input_manifest = sys.argv[1]
new_manifest_path = sys.argv[2]
ckpt = sys.argv[3]
num_sources = 1
bs = 1
data = []
with open(input_manifest, 'r') as fp:
    for line in fp:
        data.append(json.loads(line.strip()))


model = nemo_asr.models.TSEncDecCTCModelBPE.restore_from(ckpt)
# model = nemo_asr.models.EncDecCTCModelBPE.restore_from('/home/yangzhang/code/NeMo/examples/asr/asr_ctc/ngc_ckpt/2820728/Conformer-CTC-BPE/2022-04-21_20-26-13/checkpoints/Conformer-CTC-BPE.nemo')
predictions = []

pred = model.transcribe(input_manifest, batch_size=bs, num_sources=num_sources)
predictions.extend(pred)

with open(new_manifest_path, 'w', encoding='utf8') as fp:
    for data, pred in zip(data, predictions):
        data['pred_text'] = pred
        json.dump(data, fp)
        fp.write('\n')