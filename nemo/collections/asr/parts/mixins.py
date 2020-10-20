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
from abc import ABC

from omegaconf import DictConfig, OmegaConf

from nemo.collections.common import tokenizers
from nemo.utils import logging


class ASRBPEMixin(ABC):
    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        if self.tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                "`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or "
                "`wpe` for BERT based tokenizer"
            )

        if self.tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            model_path = os.path.join(self.tokenizer_dir, 'tokenizer.model')
            model_path = self.register_artifact('tokenizer.model_path', model_path)
            self.model_path = model_path

            if 'special_tokens' in self.tokenizer_cfg:
                special_tokens = self.tokenizer_cfg['special_tokens']
            else:
                special_tokens = None

            # Update special tokens
            self.tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path, special_tokens=special_tokens)

            vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path

            vocabulary = {0: '<unk>'}
            with open(vocab_path) as f:
                for i, piece in enumerate(f):
                    piece = piece.replace('\n', '')
                    vocabulary[i + 1] = piece

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            self.tokenizer.tokenizer.vocab_size = len(vocabulary)
            self.tokenizer.tokenizer.get_vocab = get_vocab
            self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            self.tokenizer_dir = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = self.tokenizer_dir

            self.tokenizer = tokenizers.AutoTokenizer(
                pretrained_model_name='bert-base-cased', vocab_file=self.tokenizer_dir, **self.tokenizer_cfg
            )

        logging.info(
            "Tokenizer {} initialized with {} tokens".format(
                self.tokenizer.__class__.__name__, self.tokenizer.vocab_size
            )
        )
