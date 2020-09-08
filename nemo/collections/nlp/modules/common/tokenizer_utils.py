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

from typing import List, Optional

import nemo
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.common.tokenizers.word_tokenizer import WordTokenizer
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import get_huggingface_pretrained_lm_models_list
from nemo.collections.nlp.modules.common.lm_utils import get_pretrained_lm_models_list
from nemo.collections.nlp.modules.common.megatron.megatron_utils import get_megatron_tokenizer

__all__ = ['get_tokenizer', 'get_tokenizer_list']


def get_tokenizer_list() -> List[str]:
    """
    Returns all all supported tokenizer names
    """
    s = set(get_pretrained_lm_models_list())
    s.update(set(get_huggingface_pretrained_lm_models_list(include_external=True)))
    return ["sentencepiece", "char", "word"] + list(s)


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    special_tokens: Optional[List[str]] = None,
):
    """
    Args:
        tokenizer_name: sentencepiece or pretrained model from the hugging face list,
            for example: bert-base-cased
            To see the list of all HuggingFace pretrained models, use: nemo_nlp.modules.common.get_huggingface_pretrained_lm_models_list()
        tokenizer_model: tokenizer model file of sentencepiece
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
    """
    full_huggingface_pretrained_model_list = get_huggingface_pretrained_lm_models_list(include_external=True)

    if tokenizer_name not in get_tokenizer_list():
        raise ValueError(
            f'Provided tokenizer_name: "{tokenizer_name}" is not supported, choose from {get_tokenizer_list()}'
        )

    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if tokenizer_name.split('-') and tokenizer_name.split('-')[0] == "megatron":
        if vocab_file is None:
            vocab_file = nemo.collections.nlp.modules.common.megatron.megatron_utils.get_megatron_vocab_file(
                tokenizer_name
            )
        tokenizer_name = get_megatron_tokenizer(tokenizer_name)

    if tokenizer_name in full_huggingface_pretrained_model_list:
        tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_name, vocab_file=vocab_file, **special_tokens_dict,)
    elif tokenizer_name == 'sentencepiece':
        tokenizer = nemo.collections.common.tokenizers.sentencepiece_tokenizer.SentencePieceTokenizer(
            model_path=tokenizer_model, special_tokens=special_tokens
        )
    elif tokenizer_name == 'word':
        return WordTokenizer(vocab_file=vocab_file, **special_tokens_dict)
    elif tokenizer_name == 'char':
        return CharTokenizer(vocab_file=vocab_file, **special_tokens_dict)

    return tokenizer
