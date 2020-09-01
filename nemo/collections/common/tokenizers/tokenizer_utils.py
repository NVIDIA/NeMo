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
from typing import List, Optional

import nemo
from nemo.collections.common.tokenizers.huggingface import (
    AlbertTokenizer,
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
)

__all__ = ['get_tokenizer']

MODEL_SPECIAL_TOKENS = {
    'bert': {
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'bos_token': '[CLS]',
        'mask_token': '[MASK]',
        'eos_token': '[SEP]',
        'cls_token': '[CLS]',
    },
    'distilbert': {
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'bos_token': '[CLS]',
        'mask_token': '[MASK]',
        'eos_token': '[SEP]',
        'cls_token': '[CLS]',
    },
    'roberta': {
        'unk_token': '<unk>',
        'sep_token': '</s>',
        'pad_token': '<pad>',
        'bos_token': '<s>',
        'mask_token': '<mask>',
        'eos_token': '</s>',
        'cls_token': '<s>',
    },
    'albert': {
        'unk_token': '<unk>',
        'sep_token': '[SEP]',
        'pad_token': '<pad>',
        'bos_token': '[CLS]',
        'mask_token': '[MASK]',
        'eos_token': '[SEP]',
        'cls_token': '[CLS]',
    },
}


TOKENIZERS = {
    'bert': BertTokenizer,
    'albert': AlbertTokenizer,
    'roberta': RobertaTokenizer,
    'distilbert': DistilBertTokenizer,
}


def get_tokenizer(
    tokenizer_name: str,
    pretrained_model_name: Optional[str] = None,
    data_file: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    sample_size: Optional[int] = None,
    special_tokens: Optional[List[str]] = None,
    vocab_file: Optional[str] = None,
    vocab_size: Optional[int] = None,
    do_lower_case: Optional[bool] = False,
):
    """
    Args:
        tokenizer_name: sentencepiece or lm model name, e.g. bert, albert
        data_file: data file used to build sentencepiece
        tokenizer_model: tokenizer model file of sentencepiece
        sample_size: sample size for building sentencepiece
        pretrained_model_name: name of the pretrained model from the hugging face list,
            for example: bert-base-cased
            To see the list of pretrained models, use: nemo_nlp.modules.common.get_pretrained_lm_models_list()
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
        vocab_size: vocab size for building sentence piece
        do_lower_case: (whether to apply lower cased) - only applicable when tokenizer is build with vocab file or with
             sentencepiece
    """
    pretrained_lm_models_list = nemo.collections.nlp.modules.common.common_utils.get_pretrained_lm_models_list()
    if pretrained_model_name and pretrained_model_name not in pretrained_lm_models_list:
        raise ValueError(
            f'Provided pretrained_model_name: "{pretrained_model_name}" is not supported, choose from {pretrained_lm_models_list}'
        )

    if tokenizer_name == "megatron":
        do_lower_case = nemo.collections.nlp.modules.common.megatron.megatron_utils.is_lower_cased_megatron(
            pretrained_model_name
        )
        vocab_file = nemo.collections.nlp.modules.common.megatron.megatron_utils.get_megatron_vocab_file(
            pretrained_model_name
        )
        tokenizer_name = 'bert'
        pretrained_model_name = None

    if tokenizer_name in ['bert', 'albert', 'roberta', 'distilbert']:
        tokenizer = get_huggingface_tokenizer(
            pretrained_model_name=pretrained_model_name,
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            tokenizer_name=tokenizer_name,
        )
    elif tokenizer_name == 'sentencepiece':
        if not tokenizer_model and not data_file:
            raise ValueError(f'either tokenizer model or data_file must passed')
        if not tokenizer_model or not os.path.exists(tokenizer_model):
            num_special_tokens = 0
            if special_tokens:
                num_special_tokens = len(set(special_tokens.values()))
            tokenizer_model, _ = nemo.collections.common.tokenizers.sentencepiece_tokenizer.create_spt_model(
                data_file=data_file,
                vocab_size=vocab_size - num_special_tokens,
                special_tokens=None,
                sample_size=sample_size,
                do_lower_case=do_lower_case,
                output_dir=os.path.dirname(data_file) + '/spt',
            )
        tokenizer = nemo.collections.common.tokenizers.sentencepiece_tokenizer.SentencePieceTokenizer(
            model_path=tokenizer_model, special_tokens=special_tokens
        )
    else:
        raise ValueError(f'{tokenizer_name} is not supported')
    return tokenizer


def get_huggingface_tokenizer(
    tokenizer_name: str,
    pretrained_model_name: Optional[str] = None,
    vocab_file: Optional[str] = None,
    do_lower_case: Optional[bool] = None,
):

    tokenizer_cls = TOKENIZERS[tokenizer_name]
    if pretrained_model_name:
        tok = tokenizer_cls.from_pretrained(pretrained_model_name)
    else:
        tok = tokenizer_cls(vocab_file=vocab_file, do_lower_case=do_lower_case)
    return tok
