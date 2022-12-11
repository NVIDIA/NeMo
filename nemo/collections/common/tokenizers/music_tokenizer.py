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

import os
from typing import Dict, List, Optional, Union

import numpy as np
from nemo.collections.common.tokenizers.music_nodes import MusicNodes

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['MusicTokenizer','create_music_vocab']

END_OF_TEXT = '<|endoftext|>'

def create_music_vocab(vocab_file, saved_json_fname):
    assert vocab_file.endswith('.txt')==True
    index=0
    json_vocab={}
    json_vocab['<|endoftext|>']=0
    index+=1
    for token in vocabulary:
        json_vocab[token]=index
        index+=1
    f=open(saved_json_fname,'w', encoding='utf-8')
    f.write(json.dumps(json_vocab,ensure_ascii=False))


class MusicTokenizer(TokenizerSpec):
    """Original GPT tokenizer for Music."""

    def __init__(self, vocab_file):
        self.tokenizer = MusicNodes(vocab_file,  errors='replace', max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']
        self.eos_id = self.eod_id
    
    def __len__(self):
        return self.vocab_size
    
    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder
    
    def text_to_tokens(self, text):
        return 'do_not_exist'
    def tokens_to_text(self, text):
        return 'do_not_exist'
    def tokens_to_ids(self, tokens):
        return 'do_not_exist'
    def ids_to_tokens(self, ids):
        return 'do_not_exist'

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def text_to_ids(self, text):
        return self.tokenizer.encode(text)

    def ids_to_text(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
