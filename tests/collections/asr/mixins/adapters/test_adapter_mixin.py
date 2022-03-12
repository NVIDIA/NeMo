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

import copy
import os
import shutil
import tempfile

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.models import configs
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.common import tokenizers
from nemo.utils.config_utils import assert_dataclass_signature_match
from nemo.collections.asr.parts.mixins.adapter_mixins import AdapterModuleMixin


class DefaultModel(torch.nn.Module, AdapterModuleMixin):

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(50, 50)

    def forward(self, x):
        ip = x
        x = self.fc(x)

        if self.is_adapter_available():
            x = x + self.adapter_layer(x)

        out = ip + x
        return out

    def num_params(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

# @pytest.fixture()
# def asr_model(test_data_dir):
#     preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
#     encoder = {
#         '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
#         'feat_in': 64,
#         'activation': 'relu',
#         'conv_mask': True,
#         'jasper': [
#             {
#                 'filters': 1024,
#                 'repeat': 1,
#                 'kernel': [1],
#                 'stride': [1],
#                 'dilation': [1],
#                 'dropout': 0.0,
#                 'residual': False,
#                 'separable': True,
#                 'se': True,
#                 'se_context_size': -1,
#             }
#         ],
#     }
#
#     decoder = {
#         '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
#         'feat_in': 1024,
#         'num_classes': -1,
#         'vocabulary': None,
#     }
#
#     tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}
#
#     modelConfig = DictConfig(
#         {
#             'preprocessor': DictConfig(preprocessor),
#             'encoder': DictConfig(encoder),
#             'decoder': DictConfig(decoder),
#             'tokenizer': DictConfig(tokenizer),
#         }
#     )
#
#     model_instance = EncDecCTCModelBPE(cfg=modelConfig)
#     return model_instance


class TestAdapterMixin:

    @pytest.mark.unit
    def test_constructor(self):
        model = DefaultModel()
        original_num_params = model.num_params()

        model.add_adapter(dim=50)
        new_num_params = model.num_params()
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_forward(self):
        torch.random.manual_seed(0)
        x = torch.randn(1, 50)

        model = DefaultModel()
        origial_output = model(x)

        model.add_adapter(dim=50)
        new_output = model(x)

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5
