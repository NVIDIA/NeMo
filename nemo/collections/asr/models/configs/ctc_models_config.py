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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from nemo.core.config import modelPT as model_cfg


@dataclass
class AudioToMelSpectrogramPreprocessorConfig:
    _target_: str = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_window_size: Optional[int] = None
    n_window_stride: Optional[int] = None
    window: str = "hann"
    normalize: str = "per_feature"
    n_fft: Optional[int] = None
    preemph: float = 0.97
    features: int = 64
    lowfreq: int = 0
    highfreq: Optional[int] = None
    log: bool = True
    log_zero_guard_type: str = "add"
    log_zero_guard_value: float = 2 ** -24
    dither: float = 1e-5
    pad_to: int = 16
    frame_splicing: int = 1
    stft_exact_pad: bool = False
    stft_conv: bool = False
    pad_value: int = 0
    mag_power: float = 2.0


@dataclass
class SpecAugmentConfig:
    _target_: str = "nemo.collections.asr.modules.SpectrogramAugmentation"
    freq_masks: int = 0
    time_masks: int = 0
    freq_width: int = 0
    time_width: Optional[Any] = 0
    rect_masks: int = 0
    rect_time: int = 0
    rect_freq: int = 0


@dataclass
class EncDecCTCDatasetConfig(model_cfg.DatasetConfig):
    manifest_filepath: Optional[str] = None
    sample_rate: int = MISSING
    labels: List[str] = MISSING
    trim_silence: bool = False

    # Tarred dataset support
    is_tarred: bool = False
    tarred_audio_filepaths: Optional[str] = None
    shuffle_n: int = 0

    # Optional
    int_values: Optional[int] = None
    augmentor: Optional[Dict[str, Any]] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    max_utts: int = 0
    blank_index: int = -1
    unk_index: int = -1
    normalize: bool = False
    trim: bool = True
    load_audio: bool = True
    parser: Optional[str] = 'en'
    add_misc: bool = False


@dataclass
class JasperEncoderConfig:
    filters: int = MISSING
    repeat: int = MISSING
    kernel: List[int] = MISSING
    stride: List[int] = MISSING
    dilation: List[int] = MISSING
    dropout: float = MISSING
    residual: bool = MISSING

    # Optional arguments
    groups: int = 1
    separable: bool = False
    heads: int = -1
    residual_mode: str = "add"
    residual_dense: bool = False
    se: bool = False
    se_reduction_ratio: int = 8
    se_context_size: int = -1
    se_interpolation_mode: str = 'nearest'
    kernel_size_factor: float = 1.0
    stride_last: bool = False


@dataclass
class ConvASREncoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASREncoder'
    jasper: Optional[JasperEncoderConfig] = field(default_factory=list)
    activation: str = MISSING
    feat_in: int = MISSING
    normalization_mode: str = "batch"
    residual_mode: str = "add"
    norm_groups: int = -1
    conv_mask: bool = True
    frame_splicing: int = 1
    init_mode: str = "xavier_uniform"


@dataclass
class ConvASRDecoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASRDecoder'
    feat_in: int = MISSING
    num_classes: int = MISSING
    init_mode: str = "xavier_uniform"
    vocabulary: Optional[List[str]] = field(default_factory=list)


@dataclass
class EncDecCTCConfig(model_cfg.ModelConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = False
    labels: List[str] = MISSING

    # Dataset configs
    train_ds: EncDecCTCDatasetConfig = EncDecCTCDatasetConfig(shuffle=True, trim_silence=True)
    validation_ds: EncDecCTCDatasetConfig = EncDecCTCDatasetConfig(shuffle=False)
    test_ds: EncDecCTCDatasetConfig = EncDecCTCDatasetConfig(manifest_filepath=None, shuffle=False)

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = model_cfg.OptimConfig(sched=model_cfg.SchedConfig())

    # Model component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = AudioToMelSpectrogramPreprocessorConfig()
    spec_augment: Optional[SpecAugmentConfig] = SpecAugmentConfig()
    encoder: Any = ConvASREncoderConfig()
    decoder: Any = ConvASRDecoderConfig()


@dataclass
class EncDecCTCModelConfig(model_cfg.ModelPTConfig):
    model: EncDecCTCConfig = EncDecCTCConfig()
