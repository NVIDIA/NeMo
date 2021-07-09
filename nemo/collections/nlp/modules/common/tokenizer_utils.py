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

import os.path
from dataclasses import MISSING, dataclass
from os import path
from typing import Dict, List, Optional

import nemo
from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelTokenizer
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.common.tokenizers.word_tokenizer import WordTokenizer
from nemo.collections.common.tokenizers.youtokentome_tokenizer import YouTokenToMeTokenizer
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import get_huggingface_pretrained_lm_models_list
from nemo.collections.nlp.modules.common.lm_utils import get_pretrained_lm_models_list
from nemo.collections.nlp.modules.common.megatron.megatron_utils import get_megatron_tokenizer
from nemo.utils import logging

__all__ = ['get_tokenizer', 'get_tokenizer_list']


def get_tokenizer_list() -> List[str]:
    """
    Returns all all supported tokenizer names
    """
    s = set(get_pretrained_lm_models_list())
    s.update(set(get_huggingface_pretrained_lm_models_list(include_external=True)))
    return ["sentencepiece", "char", "word"] + list(s)


@dataclass
class TokenizerConfig:
    library: str = MISSING
    tokenizer_model: Optional[str] = None
    vocab_size: Optional[int] = None
    vocab_file: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None
    bpe_dropout: Optional[float] = 0.0
    coverage: Optional[float] = 0.999
    training_sample_size: Optional[int] = None
    r2l: Optional[bool] = False


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast: Optional[bool] = False,
    bpe_dropout: Optional[float] = 0.0,
):
    """
    Args:
        tokenizer_name: sentencepiece or pretrained model from the hugging face list,
            for example: bert-base-cased
            To see the list of all HuggingFace pretrained models, use: nemo_nlp.modules.common.get_huggingface_pretrained_lm_models_list()
        tokenizer_model: tokenizer model file of sentencepiece or youtokentome
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
        use_fast: (only for HuggingFace AutoTokenizer) set to True to use fast HuggingFace tokenizer
        bpe_dropout: (only supported by YTTM tokenizer) BPE dropout tries to corrupt the standard segmentation procedure of BPE to help
            model better learn word compositionality and become robust to segmentation errors. 
            It has emperically been shown to improve inference time BLEU scores.
    """
    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if 'megatron' in tokenizer_name:
        if vocab_file is None:
            vocab_file = nemo.collections.nlp.modules.common.megatron.megatron_utils.get_megatron_vocab_file(
                tokenizer_name
            )
        tokenizer_name = get_megatron_tokenizer(tokenizer_name)

    if tokenizer_name == 'sentencepiece':
        return nemo.collections.common.tokenizers.sentencepiece_tokenizer.SentencePieceTokenizer(
            model_path=tokenizer_model, special_tokens=special_tokens, legacy=True
        )
    elif tokenizer_name == 'yttm':
        return YouTokenToMeTokenizer(model_path=tokenizer_model, bpe_dropout=bpe_dropout)
    elif tokenizer_name == 'word':
        return WordTokenizer(vocab_file=vocab_file, **special_tokens_dict)
    elif tokenizer_name == 'char':
        return CharTokenizer(vocab_file=vocab_file, **special_tokens_dict)

    logging.info(
        f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_name}, vocab_file: {vocab_file}, special_tokens_dict: {special_tokens_dict}, and use_fast: {use_fast}"
    )
    return AutoTokenizer(
        pretrained_model_name=tokenizer_name, vocab_file=vocab_file, **special_tokens_dict, use_fast=use_fast
    )


def get_nmt_tokenizer(
    library: str = 'yttm',
    model_name: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast: Optional[bool] = False,
    bpe_dropout: Optional[float] = 0.0,
    r2l: Optional[bool] = False,
):
    """
    Args:
        model_name: if using a pretrained model from NeMo, HuggingFace, or Megatron
        tokenizer_model: tokenizer model file of sentencepiece or youtokentome
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
        use_fast: (only for HuggingFace AutoTokenizer) set to True to use fast HuggingFace tokenizer
        bpe_dropout: (only supported by YTTM tokenizer) BPE dropout tries to corrupt the standard segmentation procedure of BPE to help
            model better learn word compositionality and become robust to segmentation errors. 
            It has emperically been shown to improve inference time BLEU scores.
        r2l: Whether to return subword IDs from right to left
    """
    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if library == 'yttm':
        logging.info(f'Getting YouTokenToMeTokenizer with model: {tokenizer_model} with r2l: {r2l}.')
        return YouTokenToMeTokenizer(model_path=tokenizer_model, bpe_dropout=bpe_dropout, r2l=r2l)
    elif library == 'huggingface':
        logging.info(f'Getting HuggingFace AutoTokenizer with pretrained_model_name: {model_name}')
        return AutoTokenizer(
            pretrained_model_name=model_name, vocab_file=vocab_file, **special_tokens_dict, use_fast=use_fast
        )
    elif library == 'sentencepiece':
        logging.info(f'Getting SentencePiece with model: {model_name}')
        return nemo.collections.common.tokenizers.sentencepiece_tokenizer.SentencePieceTokenizer(
            model_path=tokenizer_model, special_tokens=special_tokens_dict
        )
    elif library == 'byte-level':
        logging.info(f'Using byte-level tokenization')
        return ByteLevelTokenizer()
    elif library == 'megatron':
        logging.info(
            f'Getting Megatron tokenizer for pretrained model name: {model_name} and custom vocab file: {vocab_file}'
        )
        return get_tokenizer(tokenizer_name=model_name, vocab_file=vocab_file)
    else:
        raise NotImplementedError(
            'Currently we only support "yttm", "huggingface", "sentencepiece", "megatron", and "byte-level" tokenizer libraries.'
        )
