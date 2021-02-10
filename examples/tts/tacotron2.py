import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import Tacotron2Model
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


# hydra_runner is a thin NeMo wrapper around Hydra
# It looks for a config named tacotron2.yaml inside the conf folder
# Hydra parses the yaml and returns it as a Omegaconf DictConfig
@hydra_runner(config_path="conf", config_name="tacotron2")
def main(cfg):
    # Define the Lightning trainer
    trainer = pl.Trainer(**cfg.trainer)
    # exp_manager is a NeMo construct that helps with logging and checkpointing
    exp_manager(trainer, cfg.get("exp_manager", None))
    # Define the Tacotron 2 model, this will construct the model as well as
    # define the training and validation dataloaders
    model = Tacotron2Model(cfg=cfg.model, trainer=trainer)
    # Let's add a few more callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    # Call lightning trainer's fit() to train the model
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
