# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from transformers import AlbertTokenizer, BertTokenizer, RobertaTokenizer

import nemo
import nemo.collections.nlp.nm.trainables.common.huggingface as huggingface

__all__ = [
    'MODEL_SPECIAL_TOKENS',
    'MODELS',
    'get_huggingface_model',
    'TOKENIZERS',
    'get_huggingface_models_list',
    'get_sentence_piece_tokenizer',
]


def get_huggingface_model(pretrained_model_name, bert_config):
    '''
    Return the dict of special tokens associated with the model.
    Args:
    pretrained_mode_name ('str'): name of the pretrained model from the hugging face list,
        for example: bert-base-cased
        To see the list of pretrained models, call:
        nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()
        nemo_nlp.nm.trainables.huggingface.Albert.list_pretrained_models()
        nemo_nlp.nm.trainables.huggingface.Roberta.list_pretrained_models()
    bert_config: path to model configuration file.
    '''
    model_type = pretrained_model_name.split('-')[0]

    if _model_type_is_valid(model_type):
        if bert_config:
            return MODELS[model_type]['class'](config_filename=bert_config)
        else:
            return MODELS[model_type]['class'](pretrained_model_name=pretrained_model_name)
    else:
        ValueError(f'Choose pretrained model from the following list: {get_huggingface_models_list()}.')


def _model_type_is_valid(model_type):
    if model_type is None or model_type not in MODELS:
        return False
    return True


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


MODELS = {
    'bert': {'default': 'bert-base-uncased', 'class': huggingface.BERT},
    'roberta': {'defualt': 'roberta-base', 'class': huggingface.Roberta},
    'albert': {'default': 'albert-base-v2', 'class': huggingface.Albert},
}

TOKENIZERS = {'bert': BertTokenizer, 'albert': AlbertTokenizer, 'roberta': RobertaTokenizer}


def get_huggingface_models_list():
    huggingface_models = []
    for model in MODELS:
        model_names = [x.pretrained_model_name for x in MODELS[model]['class'].list_pretrained_models()]
        huggingface_models.extend(model_names)
    return huggingface_models


def get_sentence_piece_tokenizer(tokenizer_model, pretrained_model_name, special_tokens=None):
    '''
    Return SentencePieceTokenizer
    pretrained_mode_name ('str'): name of the pretrained model from the hugging face list,
        for example: bert-base-cased
        To see the list of pretrained models, call:
        nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()
        nemo_nlp.nm.trainables.huggingface.Albert.list_pretrained_models()
        nemo_nlp.nm.trainables.huggingface.Roberta.list_pretrained_models()
    special_tokens (dict): dict of special tokens (Optional)
    '''
    tokenizer = nemo.collections.nlp.data.tokenizers.SentencePieceTokenizer(model_path=tokenizer_model)

    model_type = pretrained_model_name.split('-')[0]

    if special_tokens is None:
        if _model_type_is_valid(model_type):
            special_tokens = MODEL_SPECIAL_TOKENS[model_type]
            logging.info(f'No special tokens found.')
        else:
            logging.info(f'No special tokens found for {model_type}.')

    tokenizer.add_special_tokens(special_tokens)
    return tokenizer
