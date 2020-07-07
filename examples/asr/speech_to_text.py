# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List

from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING, OmegaConf

import pytorch_lightning as pl

from nemo.collections.asr.models.ctc_models import QuartzNet
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.core.optim.novograd import Novograd

# @dataclass
# class AudioToTextDataLayer:
#     manifest_filepath: str = MISSING
#     sample_rate: int = 16000
#     labels: list = MISSING
#     batch_size: int = 64
#     trim_silence: bool = True
#     max_duration: float = 16.7
#     shuffle: bool = True

# @dataclass
# class NovogradConfig:
#     lr: float = .01

# @dataclass
# class PLpl.Trainer:
#     max_epochs: int = 5
#     gpus: int = 0

# @dataclass
# class PreprocessorConfig:
#     full_spec: str = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
#     normalize: str = "per_feature"
#     window_size: float = 0.02
#     sample_rate: int = 16000
#     window_stride: float = 0.01
#     window: str = "hann"
#     features: int = 64
#     n_fft: int = 512
#     frame_splicing: int = 1
#     dither: float = 0.00001
#     stft_conv: bool = True

# @dataclass
# class SpecAugmentConfig:
#     full_spec: str = "nemo.collections.asr.modules.SpectrogramAugmentation"
#     rect_freq: int = 50
#     rect_masks: int = 5
#     rect_time: int = 120

# @dataclass
# class SchedulerConfig:
#     monitor: str = "val_loss"
#     warmup_ratio: float = .02
#     warmup_steps: int = MISSING
#     min_lr: float = MISSING
#     last_epoch: bool = False
#     iters_per_batch: int = MISSING # computed at runtime


# defaults = [
#     {"optimizer": "novograd"}
# ]

# @dataclass
# class Config(DictConfig):
#     defaults: List[Any] = field(default_factory=lambda: defaults)
#     batch_size: int = 8
#     optimizer: Any = MISSING
#     preprocessor: PreprocessorConfig = PreprocessorConfig()
#     spec_augment: SpecAugmentConfig = SpecAugmentConfig()
#     scheduler: SchedulerConfig = SchedulerConfig()
#     PLpl.Trainer: PLpl.Trainer = PLpl.Trainer()

# cs = ConfigStore.instance()
# cs.store(group="optimizer", name="novograd", node=NovogradConfig)
# cs.store(name="config", node=Config)


#@hydra.main(config_name="config")
@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    print(cfg.pretty())
    # print(f'cfg.encoder: {cfg.encoder}')
    # print(f'cfg.decoder: {cfg.decoder}')
    # print(f'cfg.preprocessor: {cfg.preprocessor}')

    asr_model = EncDecCTCModel(
        preprocessor_config=OmegaConf.to_container(cfg.preprocessor),
        encoder_config=OmegaConf.to_container(cfg.encoder),
        decoder_config=OmegaConf.to_container(cfg.decoder),
    )
    # asr_model = EncDecCTCModel(
    #     preprocessor_config=cfg.preprocessor,
    #     encoder_config=cfg.encoder,
    #     decoder_config=cfg.decoder,
    #     spec_augment_config=cfg.spec_augment,
    # )

    asr_model.setup_training_data(cfg.AudioToTextDataLayer)
    asr_model.setup_validation_data(cfg.AudioToTextDataLayer_eval)

    # Setup optimizer and scheduler
    if cfg.pl.trainer.max_steps is None:
        if cfg.pl.trainer.gpus == 0:
            # training on CPU
            iters_per_batch = cfg.pl.trainer.max_epochs / float(cfg.pl.trainer.num_nodes * cfg.accumulate_grad_batches)
        else:
            iters_per_batch = cfg.pl.trainer.max_epochs / float(cfg.pl.trainer.gpus * cfg.pl.trainer.num_nodes * cfg.accumulate_grad_batches)
        cfg.lr_scheduler.iters_per_batch = iters_per_batch
    else:
        cfg.lr_scheduler.max_steps = cfg.pl.trainer.max_steps

    asr_model.setup_optimization(
        optim_params={
            'optimizer': "adam",
            'lr': cfg.lr,
        }
    )
            #'opt_args': [],
            # 'scheduler': CosineAnnealing, 
            # 'scheduler_args': OmegaConf.to_container(cfg.lr_scheduler)

    #trainer = pl.Trainer.from_argparse_args(OmegaConf.to_container(cfg.pl.trainer))
    trainer = pl.Trainer(**cfg.pl.trainer)
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()
