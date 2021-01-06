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


# fmt: on
def qn_15x5():
    config = [
        JasperEncoderConfig(filters=256, repeat=1, kernel=[33], stride=[2], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=1, kernel=[87], stride=[1], dilation=[2], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=1024, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False)
    ]
    return config
# fmt: on


@dataclass
class QuartzNetModelConfig(model_cfg.ModelConfig):
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

    # Model general component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = AudioToMelSpectrogramPreprocessorConfig()
    spec_augment: Optional[SpectrogramAugmentationConfig] = SpectrogramAugmentationConfig()
    encoder: ConvASREncoderConfig = ConvASREncoderConfig(activation="relu")
    decoder: ConvASRDecoderConfig = ConvASRDecoderConfig()


# Base QuartzNet class
@dataclass
class QuartzNetConfig(model_cfg.ModelPTConfig):
    model: QuartzNetModelConfig = MISSING


@dataclass
class QuartzNet15x5(QuartzNetConfig):
    # Model global arguments
    name = 'Quartznet15x5'
    model = QuartzNetModelConfig(
        spec_augment=SpectrogramAugmentationConfig(rect_masks=5, rect_freq=50, rect_time=120),
        encoder=ConvASREncoderConfig(jasper=qn_15x5(), activation="relu"),
        decoder=ConvASRDecoderConfig()
    )


class QuartzNetConfigBuilder(model_cfg.ModelPTConfigBuilder):
    VALID_CONFIGS = ['quartznet_15x5', 'quartznet_15x5_zh']

    def __init__(self, name: str = 'quartznet_15x5'):
        if name not in QuartzNetConfigBuilder.VALID_CONFIGS:
            raise ValueError("`name` must be one of : \n"
                             f"{QuartzNetConfigBuilder.VALID_CONFIGS}")

        self.name = name

        if '15x5' in name:
            model_cfg = QuartzNet15x5()
        else:
            raise ValueError("Invalid config name")

        super(QuartzNetConfigBuilder, self).__init__(model_cfg)
        self.model_cfg: QuartzNetConfig = model_cfg  # enable type hinting

    def set_labels(self, labels: List[str]):
        self.model_cfg.model.labels = labels

    def set_separable(self, separable: bool):
        self.model_cfg.model.separable = separable

    def set_repeat(self, repeat: int):
        self.model_cfg.model.repeat = repeat

    def set_sample_rate(self, sample_rate: int):
        self.model_cfg.model.sample_rate = sample_rate

    def set_dropout(self, dropout: float = 0.0):
        self.model_cfg.model.dropout = dropout

    # Note: Autocomplete for users wont work without these overrides
    # But practically it is not needed since python will infer at runtime

    # def set_train_ds(self, cfg: Optional[ctc_cfg.EncDecCTCDatasetConfig] = None):
    #     super(QuartzNetConfigBuilder, self).set_train_ds(cfg)
    #
    # def set_validation_ds(self, cfg: Optional[ctc_cfg.EncDecCTCDatasetConfig] = None):
    #     super(QuartzNetConfigBuilder, self).set_validation_ds(cfg)
    #
    # def set_test_ds(self, cfg: Optional[ctc_cfg.EncDecCTCDatasetConfig] = None):
    #     super(QuartzNetConfigBuilder, self).set_test_ds(cfg)

    def _finalize_cfg(self):
        # propagate labels 
        self.model_cfg.model.train_ds.labels = self.model_cfg.model.labels
        self.model_cfg.model.validation_ds.labels = self.model_cfg.model.labels
        self.model_cfg.model.test_ds.labels = self.model_cfg.model.labels
        self.model_cfg.model.decoder.vocabulary = self.model_cfg.model.labels

        # propagate num classes
        self.model_cfg.model.decoder.num_classes = len(self.model_cfg.model.labels)

        # propagate sample rate
        self.model_cfg.model.sample_rate = self.model_cfg.model.sample_rate
        self.model_cfg.model.preprocessor.sample_rate = self.model_cfg.model.sample_rate
        self.model_cfg.model.train_ds.sample_rate = self.model_cfg.model.sample_rate
        self.model_cfg.model.validation_ds.sample_rate = self.model_cfg.model.sample_rate
        self.model_cfg.model.test_ds.sample_rate = self.model_cfg.model.sample_rate

        # propagate filters
        self.model_cfg.model.encoder.feat_in = self.model_cfg.model.preprocessor.features
        self.model_cfg.model.decoder.feat_in = self.model_cfg.model.encoder.jasper[-1].filters

        # propagate separable
        for layer in self.model_cfg.model.encoder.jasper[:-1]:  # type: JasperEncoderConfig
            layer.separable = self.model_cfg.model.separable

        # propagate repeat
        for layer in self.model_cfg.model.encoder.jasper[1:-2]:  # type: JasperEncoderConfig
            layer.repeat = self.model_cfg.model.repeat

        # propagate dropout
        for layer in self.model_cfg.model.encoder.jasper:  # type: JasperEncoderConfig
            layer.dropout = self.model_cfg.model.dropout

    def build(self) -> QuartzNetConfig:
        return super(QuartzNetConfigBuilder, self).build()


if __name__ == '__main__':

    cfg = QuartzNet15x5()
    print(cfg)
