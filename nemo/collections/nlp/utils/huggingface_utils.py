# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

__all__ = ['MODEL_SPECIAL_TOKENS', 'MODEL_NAMES']

MODEL_SPECIAL_TOKENS = {
    "bert": {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "bos_token": "[CLS]",
        "mask_token": "[MASK]",
        "eos_token": "[SEP]",
        "cls_token": "[CLS]",
    },
    "roberta": {
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "mask_token": "<mask>",
        "eos_token": "</s>",
        "cls_token": "<s>",
    },
    "albert": {
        "unk_token": "<unk>",
        "sep_token": "[SEP]",
        "pad_token": "<pad>",
        "bos_token": "[CLS]",
        "mask_token": "[MASK]",
        "eos_token": "[SEP]",
        "cls_token": "[CLS]",
    },
}


MODEL_NAMES = {
    "bert": {"model_name": "bert-base-uncased", "tokenizer_name": "bert-base-uncased",},
    "roberta": {"model_name": "roberta-base", "tokenizer_name": "roberta-base",},
    "albert": {"model_name": "albert-base-v2", "tokenizer_name": "albert-base-v2",},
}
