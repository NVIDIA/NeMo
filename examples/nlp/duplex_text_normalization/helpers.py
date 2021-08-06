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

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.models import DuplexDecoderModel, DuplexTaggerModel
from nemo.utils import logging

__all__ = ['TAGGER_MODEL', 'DECODER_MODEL', 'MODEL_NAMES', 'instantiate_model_and_trainer']

TAGGER_MODEL = 'tagger'
DECODER_MODEL = 'decoder'
MODEL_NAMES = [TAGGER_MODEL, DECODER_MODEL]


def instantiate_model_and_trainer(cfg: DictConfig, model_name: str, do_training: bool):
    """ Function for instantiating a model and a trainer
    Args:
        cfg: The config used to instantiate the model and the trainer.
        model_name: A str indicates whether the model to be instantiated is a tagger or a decoder (i.e., model_name should be either TAGGER_MODEL or DECODER_MODEL).
        do_training: A boolean flag indicates whether the model will be trained or evaluated.

    Returns:
        trainer: A PyTorch Lightning trainer
        model: A NLPModel that can either be a DuplexTaggerModel or a DuplexDecoderModel
    """
    assert model_name in MODEL_NAMES
    logging.info(f'Model {model_name}')

    # Get configs for the corresponding models
    trainer_cfg = cfg.get(f'{model_name}_trainer')
    model_cfg = cfg.get(f'{model_name}_model')
    pretrained_cfg = cfg.get(f'{model_name}_pretrained_model', None)

    trainer = pl.Trainer(**trainer_cfg)

    if not pretrained_cfg:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        if model_name == TAGGER_MODEL:
            model = DuplexTaggerModel(model_cfg, trainer=trainer)
        if model_name == DECODER_MODEL:
            model = DuplexDecoderModel(model_cfg, trainer=trainer)
    else:
        logging.info(f'Loading pretrained model {pretrained_cfg}')
        if model_name == TAGGER_MODEL:
            model = DuplexTaggerModel.restore_from(pretrained_cfg)
        if model_name == DECODER_MODEL:
            model = DuplexDecoderModel.restore_from(pretrained_cfg)

    # Set model.lang (if it is still None)
    if model.lang is None:
        model.lang = cfg.lang
    assert model.lang in constants.SUPPORTED_LANGS
    # Setup covering grammars (if enabled)
    # We only support integrating with English TN covering grammars at the moment
    if model_name == DECODER_MODEL and model_cfg.use_cg and cfg.lang == constants.ENGLISH:
        if model.cg_normalizer is None:
            model.setup_cgs(model_cfg)

    # Setup train and validation data
    if do_training:
        model.setup_training_data(train_data_config=cfg.data.train_ds)
        model.setup_validation_data(val_data_config=cfg.data.validation_ds)

    logging.info(f'Model Device {model.device}')
    return trainer, model


def flatten(l):
    """ flatten a list of lists """
    return [item for sublist in l for item in sublist]
