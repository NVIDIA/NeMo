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


from typing import Dict

import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import build_training_sample

class GPTRequestDataset(Dataset):
    def __init__(self, request: Dict, tokenizer) -> None:
        super().__init__()
        self.request = request
        self.tokenizer = tokenizer

        # tokenize prompt
        self.request['tokenized_prompt'] = self.tokenizer.text_to_tokens(request['prompt'])
        tokens = self.tokenizer.text_to_ids(request['prompt'])
        self.request['tokens'] = torch.tensor(tokens)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request

class T5RequestDataset(Dataset):
    def __init__(self, request: Dict, tokenizer) -> None:
        super().__init__()
        self.request = request
        self.tokenizer = tokenizer

        # tokenize prompt
        self.request['tokenized_prompt'] = ' '.join(self.tokenizer.text_to_tokens(request['prompt']))
        tokens = self.tokenizer.text_to_ids(request['prompt'])
        self.request['tokens'] = torch.tensor(tokens)
        self.mask_prompt(self.request['prompt'])

    def mask_prompt(self, sample):
        if '<mask>' not in sample:
            raise ValueError(f"Did not find any <mask> tokens in prompt {sample}.")
        
        sample = sample.split()
        sentinel_idx = 0
        for i, word in enumerate(sample):
            if word == '<mask>':
                sample[i] = f'<extra_id_{sentinel_idx}>'
                sentinel_idx += 1
        sample = ' '.join(sample)
        sample = torch.LongTensor(self.tokenizer.text_to_ids(sample))
        self.request['masked_sample'] = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request
