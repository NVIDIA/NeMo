import os
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from utils import TAGGER_MODEL, DECODER_MODEL, initialize_model_and_trainer

from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.collections.nlp.models import NeuralTextNormalizationModel

@hydra_runner(config_path="conf", config_name="text_normalization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    tagger_trainer, tagger_model = initialize_model_and_trainer(cfg, TAGGER_MODEL)
    decoder_trainer, decoder_model = initialize_model_and_trainer(cfg, DECODER_MODEL)
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

if __name__ == '__main__':
    main()
