from pytorch_lightning import Trainer

from pytorch_lightning import Trainer

from nemo.collections.multimodal.models.text_to_image.controlnet import ImageLogger
from nemo.collections.multimodal.models.text_to_image.controlnet.controlnet import MegatronControlNet
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


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
