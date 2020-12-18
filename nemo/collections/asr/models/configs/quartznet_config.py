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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import ConvASRDecoderConfig, ConvASREncoderConfig, JasperEncoderConfig
from nemo.core.config import modelPT as model_cfg
from nemo.collections.asr.models.configs import ctc_models_config as ctc_cfg


def qn_15x5(separable, dropout):
    config = [
        JasperEncoderConfig(
            filters=256,
            repeat=1,
            kernel=[33],
            stride=[2],
            separable=separable,
            dilation=[1],
            dropout=dropout,
            residual=False,
        ),
        JasperEncoderConfig(
            filters=256,
            repeat=1,
            kernel=[33],
            stride=[1],
            separable=separable,
            dilation=[1],
            dropout=dropout,
            residual=True,
        ),
        # ... repeat 14 more times
        JasperEncoderConfig(
            filters=1024, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=dropout, residual=False,
        ),
    ]
    return config


@dataclass
class QuartzNetConfig(model_cfg.ModelConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = True
    labels: List[str] = MISSING

    # Dataset configs
    train_ds: ctc_cfg.EncDecCTCDatasetConfig = ctc_cfg.EncDecCTCDatasetConfig(
        manifest_filepath=None, shuffle=True, trim_silence=True
    )
    validation_ds: ctc_cfg.EncDecCTCDatasetConfig = ctc_cfg.EncDecCTCDatasetConfig(
        manifest_filepath=None, shuffle=False
    )
    test_ds: ctc_cfg.EncDecCTCDatasetConfig = ctc_cfg.EncDecCTCDatasetConfig(manifest_filepath=None, shuffle=False)

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = model_cfg.OptimConfig(sched=model_cfg.SchedConfig())

    # Model component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = AudioToMelSpectrogramPreprocessorConfig()
    spec_augment: Optional[SpectrogramAugmentationConfig] = SpectrogramAugmentationConfig()
    encoder: Any = ConvASREncoderConfig(activation="relu")
    decoder: Any = ConvASRDecoderConfig()


@dataclass
class QuartzNet15x5(QuartzNetConfig):
    # Model global arguments
    labels: List[str] = MISSING

    # Model component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = AudioToMelSpectrogramPreprocessorConfig()
    spec_augment: Optional[SpectrogramAugmentationConfig] = SpectrogramAugmentationConfig(
        rect_masks=5, rect_freq=50, rect_time=120
    )
    encoder: Any = ConvASREncoderConfig(activation="relu")
    decoder: Any = ConvASRDecoderConfig()


def get_qn_15x5_config(labels=MISSING, ):
    pass


if __name__ == '__main__':

    cfg = QuartzNet15x5()
    print(cfg)
