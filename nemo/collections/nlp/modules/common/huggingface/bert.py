# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from transformers import BertModel

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.classes import typecheck

__all__ = ['BertEncoder']


class BertEncoder(BertModel, BertModule):
    """
    Wraps around the Huggingface transformers implementation repository for easy use within NeMo.
    """

    @typecheck()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        res = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        return res
