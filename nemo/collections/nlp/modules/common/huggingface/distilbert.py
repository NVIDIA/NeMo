# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 The Google AI Language Team Authors and
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

from transformers import DistilBertModel

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.classes import typecheck
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['DistilBertEncoder']


@experimental
class DistilBertEncoder(DistilBertModel, BertModule):
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
        # distilBert does not use token_type_ids as the most of the other Bert models
        if 'token_type_ids' in kwargs:
            logging.info("DistilBert doesn’t use token_type_ids, and it is ignored.")
            del kwargs['token_type_ids']

        res = super().forward(**kwargs)[0]
        return res
