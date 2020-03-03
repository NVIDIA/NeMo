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
import os

from transformers import AlbertTokenizer, BertTokenizer, RobertaTokenizer

import nemo

__all__ = ['MODEL_SPECIAL_TOKENS', 'TOKENIZERS', 'get_tokenizer', 'get_bert_special_tokens']

logging = nemo.logging

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


TOKENIZERS = {'bert': BertTokenizer, 'albert': AlbertTokenizer, 'roberta': RobertaTokenizer}


def get_bert_special_tokens(bert_derivative):
    return MODEL_SPECIAL_TOKENS[bert_derivative]


def get_tokenizer(tokenizer_name, pretrained_model_name, tokenizer_model=None, special_tokens=None):
    '''
    Args:
    tokenizer_name: sentencepiece or nemobert
    pretrained_mode_name ('str'): name of the pretrained model from the hugging face list,
        for example: bert-base-cased
        To see the list of pretrained models, use: nemo_nlp.nm.trainables.get_bert_models_list()
    tokenizer_model (path): only used for sentencepiece tokenizer
    special_tokens (dict): dict of special tokens (Optional)
    '''

    if tokenizer_name == 'nemobert':
        tokenizer = nemo.collections.nlp.data.tokenizers.NemoBertTokenizer(pretrained_model=pretrained_model_name)
    elif tokenizer_name == 'sentencepiece':
        if not os.path.exists(tokenizer_model):
            raise FileNotFoundError(f'{tokenizer_model} tokenizer model not found')

        tokenizer = nemo.collections.nlp.data.tokenizers.SentencePieceTokenizer(model_path=tokenizer_model)
        model_type = pretrained_model_name.split('-')[0]
        if special_tokens is None:
            if model_type not in MODEL_SPECIAL_TOKENS:
                logging.info(f'No special tokens found for {model_type}.')
            else:
                special_tokens = MODEL_SPECIAL_TOKENS[model_type]
        tokenizer.add_special_tokens(special_tokens)
    else:
        raise ValueError(f'{tokenizer_name} is not supported')
    return tokenizer
