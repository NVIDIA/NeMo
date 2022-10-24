from pathlib import Path

import pytorch_lightning as pl

pl.seed_everything(0)

from omegaconf import OmegaConf
from src.generate_data import generate_dataset
from src.multi_classification_models import EncDecMultiClassificationModel

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="./configs", config_name="debug")
def main(cfg):

    data_cfg = cfg.data
    if not data_cfg.skip:
        generate_dataset(
            "synth_audio_train",
            data_cfg.num_samples,
            data_cfg.sample_duration,
            data_cfg.total_duration,
            data_cfg.sample_rate,
        )

        generate_dataset(
            "synth_audio_val", 500, data_cfg.sample_duration, data_cfg.total_duration, data_cfg.sample_rate,
        )

    if not data_cfg.data_only:
        OmegaConf.set_struct(cfg, False)
        cfg.model.train_ds.manifest_filepath = str(Path("synth_audio_train") / Path("synth_manifest.json"))
        cfg.model.validation_ds.manifest_filepath = str(Path("synth_audio_val") / Path("synth_manifest.json"))

        if "augmentor" in cfg.model.train_ds and "noise" in cfg.model.train_ds.augmentor:
            cfg.model.train_ds.augmentor.noise.manifest_path = str(
                Path("synth_audio_val") / Path("synth_manifest.json")
            )
        OmegaConf.set_struct(cfg, True)

        logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))
        asr_model = EncDecMultiClassificationModel(cfg=cfg.model, trainer=trainer)

        trainer.fit(asr_model)

        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            if asr_model.prepare_test(trainer):
                trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
