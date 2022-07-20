
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import PunctCapSegModel
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_config import (
    PunctuationCapitalizationConfig,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="punct_cap_seg")
def main(cfg: DictConfig) -> None:
    # torch.manual_seed(42)
    print(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    model = PunctCapSegModel(cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main()
