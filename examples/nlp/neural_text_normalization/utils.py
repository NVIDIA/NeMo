import os
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import (
    TextNormalizationTaggerModel,
    TextNormalizationDecoderModel,
    NeuralTextNormalizationModel
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

TAGGER_MODEL  = 'tagger'
DECODER_MODEL = 'decoder'
MODEL_NAMES   = [TAGGER_MODEL, DECODER_MODEL]

def initialize_model_and_trainer(cfg: DictConfig, model_name: str):
    assert(model_name in MODEL_NAMES)
    logging.info(f'Model {model_name}')

    # Get configs for the corresponding models
    trainer_cfg     = cfg.get(f'{model_name}_trainer')
    model_cfg       = cfg.get(f'{model_name}_model')
    pretrained_cfg  = cfg.get(f'{model_name}_pretrained_model', None)

    trainer = pl.Trainer(**trainer_cfg)

    if not pretrained_cfg:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        if model_name == TAGGER_MODEL:
            model = TextNormalizationTaggerModel(model_cfg, trainer=trainer)
        if model_name == DECODER_MODEL:
            model = TextNormalizationDecoderModel(model_cfg, trainer=trainer)
    else:
        logging.info(f'Loading pretrained model {pretrained_cfg}')
        if model_name == TAGGER_MODEL:
            model = TextNormalizationTaggerModel.from_pretrained(pretrained_cfg)
        if model_name == DECODER_MODEL:
            model = TextNormalizationDecoderModel.from_pretrained(pretrained_cfg)

    logging.info(f'Model Device {model.device}')
    return trainer, model
