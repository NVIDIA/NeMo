from typing import Optional

from nemo_run import Config, cli
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from nemo import lightning as nl


def tensorboard_logger(name: str, save_dir: str = "tb_logs") -> Config[TensorBoardLogger]:
    return Config(TensorBoardLogger, save_dir=save_dir, name=name)


def wandb_logger(project: str, name: str, entity: Optional[str] = None) -> Config[WandbLogger]:
    cfg = Config(
        WandbLogger,
        project=project,
        name=name,
        config={},
    )

    if entity:
        cfg.entity = entity

    return cfg


@cli.factory(is_target_default=True)
def default_log(
    dir: Optional[str] = None,
    name: str = "default",
    tensorboard_logger: Optional[Config[TensorBoardLogger]] = None,
    wandb_logger: Optional[Config[WandbLogger]] = None,
) -> Config[nl.NeMoLogger]:
    ckpt = Config(
        nl.ModelCheckpoint,
        save_best_model=False,
        save_last=True,
        save_top_k=10,
        every_n_train_steps=200,
        filename="{model_name}--{val_loss:.2f}-{step}-{consumed_samples}",
    )

    return Config(
        nl.NeMoLogger,
        ckpt=ckpt,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        dir=dir,
    )


@cli.factory(is_target_default=True)
def default_resume(resume_if_exists=True, resume_ignore_no_checkpoint=True) -> Config[nl.AutoResume]:
    return Config(
        nl.AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
    )
