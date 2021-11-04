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
from typing import Any, Dict, Optional, Tuple

from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    PunctuationCapitalizationDataConfig
)
from nemo.core.config.modelPT import OptimConfig, SchedConfig


@dataclass
class MTSchedConfig(SchedConfig):
    name: str = 'InverseSquareRootAnnealing'
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1


# TODO: Refactor this dataclass to to support more optimizers (it pins the optimizer to Adam-like optimizers).
@dataclass
class MTOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    sched: Optional[MTSchedConfig] = MTSchedConfig()


@dataclass
class LanguageModelConfig:
    pretrained_model_name: str = MISSING
    config_file: Optional[str] = None
    config: Optional[Dict] = None
    lm_checkpoint: Optional[str] = None


@dataclass
class PunctHeadConfig:
    punct_num_fc_layers: int = 1
    fc_dropout: float = 0.1
    activation: str = 'relu'
    use_transformer_init: bool = True


@dataclass
class CapitHeadConfig:
    capit_num_fc_layers: int = 1
    fc_dropout: float = 0.1
    activation: str = 'relu'
    use_transformer_init: bool = True


@dataclass
class ClassLabels:
    punct_labels_file: Optional[str] = "punct_label_ids.csv"
    capit_labels_file: Optional[str] = "capit_label_ids.csv"


@dataclass
class CommonDatasetParameters:
    data_dir: Optional[str] = None
    max_seq_length: Optional[int] = 128
    pad_label: str = MISSING
    ignore_extra_tokens: bool = False
    ignore_start_end: bool = True
    use_cache: bool = True
    num_workers: Optional[int] = 2
    pin_memory: bool = False
    drop_last: bool = False


@dataclass
class PunctuationCapitalizationModelConfig:
    dataset: Optional[CommonDatasetParameters] = CommonDatasetParameters()
    train_ds: Optional[PunctuationCapitalizationDataConfig] = PunctuationCapitalizationDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        metadata_file=MISSING,
    )
    validation_ds: Optional[PunctuationCapitalizationDataConfig] = PunctuationCapitalizationDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        metadata_file=MISSING,
    )
    test_ds: Optional[PunctuationCapitalizationDataConfig] = PunctuationCapitalizationDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        metadata_file=MISSING,
    )
    punct_label_ids: Optional[Dict[str, int]] = None
    capit_label_ids: Optional[Dict[str, int]] = None
    class_labels: Optional[ClassLabels] = ClassLabels()

    punct_head: PunctHeadConfig = PunctHeadConfig()
    capit_head: CapitHeadConfig = CapitHeadConfig()

    tokenizer: Any = MISSING

    shuffle_train_dataset: bool = False

    language_model: LanguageModelConfig = LanguageModelConfig()

    optim: Optional[OptimConfig] = MTOptimConfig()
