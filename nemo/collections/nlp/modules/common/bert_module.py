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

import re
from typing import Dict, Optional

import torch

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['BertModule']


@experimental
class BertModule(NeuralModule):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @classmethod
    def restore_from(cls, restore_path: str):
        """Restores module/model with weights"""
        pass

    @classmethod
    def save_to(self, save_path: str):
        """Saves module/model with weights"""
        pass

    def restore_weights(self, restore_path: str):
        """Restores module/model's weights"""
        logging.info(f"restore from {restore_path}")
        pretrained_dict = torch.load(restore_path)

        # backward compatibility with NeMo0.11
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["state_dict"]

        # remove prefix from pretrained dict
        m = re.match("^bert.+?\.", list(pretrained_dict.keys())[0])
        if m:
            prefix = m.group(0)
            pretrained_dict = {k[len(prefix) :]: v for k, v in pretrained_dict.items()}
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert len(pretrained_dict) == len(model_dict)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
