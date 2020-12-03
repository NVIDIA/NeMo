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
from typing import Optional

from transformers import AutoTokenizer as AUTOTOKENIZER

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = [
    'AutoTokenizer',
]


def handle_quotes(text):
    text_ = ""
    quote = 0
    i = 0
    while i < len(text):
        if text[i] == "\"":
            if quote % 2:
                text_ = text_[:-1] + "\""
            else:
                text_ += "\""
                i += 1
            quote += 1
        else:
            text_ += text[i]
        i += 1
    return text_


def remove_spaces(text):
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace("[ ", "[")
    text = text.replace(" ]", "]")
    text = text.replace(" / ", "/")
    text = text.replace("„ ", "„")
    text = text.replace(" - ", "-")
    text = text.replace(" ' ", "'")
    text = re.sub(r'([0-9])( )([\.,])', '\\1\\3', text)
    text = re.sub(r'([\.,])( )([0-9])', '\\1\\3', text)
    text = re.sub(r'([0-9])(:)( )([0-9])', '\\1\\2\\4', text)
    text = text.replace(" %", "%")
    text = text.replace("$ ", "$")
    text = re.sub(r'([^0-9])(,)([0-9])', '\\1\\2 \\3', text)
    return text


class AutoTokenizer(TokenizerSpec):
    '''
        Wrapper of HuggingFace AutoTokenizer https://huggingface.co/transformers/model_doc/auto.html#autotokenizer.
    '''

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_file: Optional[str] = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        use_fast: Optional[bool] = False,
    ):

        """
        Args:
            pretrained_model_name: corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument. 
                For more details please refer to https://huggingface.co/transformers/_modules/transformers/tokenization_auto.html#AutoTokenizer.from_pretrained. 
                The list of all supported models can be found here: ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
            vocab_file: path to file with vocabulary which consists
                of characters separated by '\n'.
            mask_token: mask token 
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            use_fast: whether to use fast HuggingFace tokenizer
        """
        try:
            if vocab_file is not None:
                message = 'Using "slow" HuggingFace tokenizer'
                if use_fast:
                    message += f'{vocab_file} is ignored in "fast" tokenizers, using a "slow" version'
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name, vocab_file=vocab_file, use_fast=False
                )
            else:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name, use_fast=use_fast
                )
        except Exception as e:
            raise ValueError(f'{pretrained_model_name} is not supported by HuggingFace. {e}')

        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file will be added to 
            the vocabulary and, as a result, the model should be resized, for example:
            
            # define your model
            pretrained_model_name = 'roberta-base'
            model = nemo_nlp.modules.get_lm_model(pretrained_model_name=pretrained_model_name)
            
            # define pretrained tokenizer
            tokenizer_default = nemo_nlp.modules.get_tokenizer(tokenizer_name=pretrained_model_name)
            
            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer_default.add_special_tokens(special_tokens_dict=special_tokens)
            
            # resize your model so that the embeddings for newly added tokens are updated during training/finetuning
            model.resize_token_embeddings(tokenizer_default.vocab_size)
            
            See NLP_Tokenizers.ipynb for more details.
            """
            logging.warning(
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly, '
                f'see NLP_Tokenizers.ipynb for more details.'
            )
        self.add_special_tokens(special_tokens_dict)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
            Tokens are only added if they are not already in the vocabulary.
        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            logging.info(f'{num_tokens_added} special tokens added, resize your model accordingly.')
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return remove_spaces(handle_quotes(text.strip()))

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        text = self.tokens_to_text(tokens_clean)
        return text

    @property
    def pad_id(self):
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]

    @property
    def unk_id(self):
        return self.tokens_to_ids([getattr(self, 'unk_token')])[0]

    @property
    def mask_id(self):
        return self.tokens_to_ids([getattr(self, 'mask_token')])[0]

    @property
    def name(self):
        return type(self.tokenizer).__name__
