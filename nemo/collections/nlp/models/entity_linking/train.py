"""
## Task

## Preparing the dataset

## Model Training
"""


import os

from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from self_alignment_pretraining_model_no_pretokenization import SelfAlignmentPretrainingModel

#TODO: Change config location to /conf to match other nemo exaples
@hydra_runner(config_path="../configs", config_name="full_biomegatron_sap_config1.yaml")
def main(cfg: DictConfig) -> None:
    logging.info(f"\nConfig Params:\n{cfg.pretty()}")
    trainer = Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f"Loading weights from pretrained model {cfg.model.language_model.pretrained_model_name}")
    model = SelfAlignmentPretrainingModel(cfg=cfg.model, trainer=trainer)
    logging.info("===========================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')
    logging.info("===========================================================================================")

    if cfg.model.nemo_path:
        # '.nemo' file contains the last checkpoint and the params to initialize the model
        model.save_to(cfg.model.nemo_path)
        logging.info(f'Model is saved into `.nemo` file: {cfg.model.nemo_path}')

if __name__ == '__main__':
    main()
