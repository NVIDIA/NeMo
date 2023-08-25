from datetime import timedelta

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.strategies.ddp import DDPStrategy

from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)
from nemo.collections.multimodal.models.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.controlnet.util import ImageLogger
from nemo.collections.multimodal.parts.stable_diffusion.utils import instantiate_from_config
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import StatelessTimer, exp_manager


class MegatronControlNetTrainerBuilder(MegatronTrainerBuilder):
    """Builder for T5 model Trainer with overrides."""

    def create_trainer(self, callbacks=[]) -> Trainer:
        strategy = self._training_strategy()
        plugins = self._plugins()
        return Trainer(plugins=plugins, strategy=strategy, **self.cfg.trainer, callbacks=callbacks)


@hydra_runner(config_path='conf', config_name='controlnet_v1-5.yaml')
def main(cfg):
    callbacks = []

    if cfg.model.get('image_logger', None):
        callbacks.append(ImageLogger(**cfg.model.image_logger))

    trainer = MegatronControlNetTrainerBuilder(cfg).create_trainer(callbacks=callbacks)

    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MegatronControlNet(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
