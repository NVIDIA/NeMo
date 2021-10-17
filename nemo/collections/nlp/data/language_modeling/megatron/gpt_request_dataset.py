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
        '''
        enc_ids = self.tokenizer.ids_to_tokens(self.request['training_sample']['text_enc'])
        dec_ids = self.tokenizer.ids_to_tokens(self.request['training_sample']['text_dec'])
        import ipdb; ipdb.set_trace()
        '''

    def mask_prompt(self, sample):
        sample = [self.tokenizer.text_to_ids(sample)]
        training_sample = build_training_sample(
            sample=sample,
            target_seq_length=len(sample[0]),
            max_seq_length=len(sample[0]),
            max_seq_length_dec=128,
            vocab_id_list=self.tokenizer.vocab,
            vocab_id_to_token_dict={idx: token for idx, token in enumerate(self.tokenizer.vocab)},
            cls_id=self.tokenizer.cls_id,
            sep_id=self.tokenizer.sep_id,
            mask_id=self.tokenizer.mask_id,
            pad_id=self.tokenizer.pad_id,
            masked_lm_prob=0.15,
            np_rng=np.random.RandomState(seed=1337),
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            sentinel_tokens=self.tokenizer.tokenizer.additional_special_tokens_ids,
        )
        self.request['training_sample'] = training_sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request
