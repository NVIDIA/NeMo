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

@hydra_runner(config_path="conf", config_name="text_normalization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    tagger_trainer, tagger_model = initialize(cfg, TAGGER_MODEL)
    decoder_trainer, decoder_model = initialize(cfg, DECODER_MODEL)
    tn_model = NeuralTextNormalizationModel(tagger_model, decoder_model)

    # Train the tagger
    logging.info("================================================================================================")
    logging.info('Starting training tagger...')
    tagger_trainer.fit(tagger_model)
    logging.info('Training finished!')

    # Train the decoder
    logging.info("================================================================================================")
    logging.info('Starting training decoder...')
    decoder_trainer.fit(decoder_model)
    logging.info('Training finished!')

def initialize(cfg: DictConfig, model_name: str):
    assert(model_name in MODEL_NAMES)
    logging.info(f'Model {model_name}')

    # Get configs for the corresponding models
    trainer_cfg     = cfg.get(f'{model_name}_trainer')
    model_cfg       = cfg.get(f'{model_name}_model')
    pretrained_cfg  = cfg.get(f'{model_name}_pretrained_model')
    exp_manager_cfg = f'{model_name}_exp_manager'

    trainer = pl.Trainer(**trainer_cfg)
    exp_dir = exp_manager(trainer, cfg.get(exp_manager_cfg, None))

    if not cfg.pretrained_model:
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

if __name__ == '__main__':
    main()
