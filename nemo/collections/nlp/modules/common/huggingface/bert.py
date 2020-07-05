# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from transformers import BertModel

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.classes import typecheck
from nemo.utils.decorators import experimental

__all__ = ['BertEncoder']


@experimental
class BertEncoder(BertModel, BertModule):
    """
    Wraps around the Huggingface transformers implementation repository for easy use within NeMo.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @typecheck()
    def forward(self, **kwargs):
        res = super().forward(**kwargs)[0]
        return res
