import pytorch_lightning as pl

from nemo.collections.tts.models import GlowTTSModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="glow_tts_config")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    glow_tts_model = GlowTTSModel(cfg=cfg.model, trainer=trainer)

    trainer.fit(glow_tts_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
