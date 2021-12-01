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


from typing import List

import torch
from torch.utils.data.dataset import Dataset


class GPTRequestDataset(Dataset):
    def __init__(self, requests: List, tokenizer, tokens_to_generate: int) -> None:
        super().__init__()
        self.requests = requests
        self.tokenizer = tokenizer
        self.tokens_to_generate = tokens_to_generate
        self.tokens = []

        # tokenize prompt
        for request in self.requests:
            self.tokens.append(torch.tensor(self.tokenizer.text_to_ids(request)))

        self.data = {
            'data': self.tokens,
            'tokens_to_generate': self.tokens_to_generate,
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data
