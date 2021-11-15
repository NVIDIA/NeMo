# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from omegaconf.omegaconf import DictConfig, MISSING, OmegaConf

from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    PunctuationCapitalizationEvalDataConfig,
    PunctuationCapitalizationTrainDataConfig,
    legacy_data_config_to_new_data_config,
)
from nemo.core.config import TrainerConfig
from nemo.core.config.modelPT import NemoConfig, OptimConfig, SchedConfig
from nemo.utils.exp_manager import ExpManagerConfig


DEFAULT_IGNORE_EXTRA_TOKENS = False
DEFAULT_IGNORE_START_END = True


@dataclass
class PunctuationCapitalizationSchedConfig(SchedConfig):
    name: str = 'InverseSquareRootAnnealing'
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1


# TODO: Refactor this dataclass to to support more optimizers (it pins the optimizer to Adam-like optimizers).
@dataclass
class PunctuationCapitalizationOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    sched: Optional[PunctuationCapitalizationSchedConfig] = PunctuationCapitalizationSchedConfig()


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
class CommonDatasetParameters:
    pad_label: str = MISSING
    ignore_extra_tokens: bool = DEFAULT_IGNORE_EXTRA_TOKENS
    ignore_start_end: bool = DEFAULT_IGNORE_START_END
    punct_label_ids: Optional[Dict[str, int]] = None
    punct_label_vocab_file: Optional[str] = None
    capit_label_ids: Optional[Dict[str, int]] = None
    capit_label_vocab_file: Optional[str] = None


@dataclass
class PunctuationCapitalizationModelConfig:
    common_dataset_parameters: Optional[CommonDatasetParameters] = CommonDatasetParameters()
    train_ds: Optional[PunctuationCapitalizationTrainDataConfig] = PunctuationCapitalizationTrainDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        tar_metadata_file=MISSING,
        tokens_in_batch=MISSING,
    )
    validation_ds: Optional[PunctuationCapitalizationEvalDataConfig] = PunctuationCapitalizationEvalDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        tar_metadata_file=MISSING,
        tokens_in_batch=MISSING,
    )
    test_ds: Optional[PunctuationCapitalizationEvalDataConfig] = PunctuationCapitalizationEvalDataConfig(
        text_file=MISSING,
        labels_file=MISSING,
        use_tarred_dataset=MISSING,
        tar_metadata_file=MISSING,
        tokens_in_batch=MISSING,
    )

    punct_head: PunctHeadConfig = PunctHeadConfig()
    capit_head: CapitHeadConfig = CapitHeadConfig()

    tokenizer: Any = MISSING

    language_model: LanguageModelConfig = LanguageModelConfig()

    optim: Optional[OptimConfig] = PunctuationCapitalizationOptimConfig()


@dataclass
class PunctuationCapitalizationConfig(NemoConfig):
    pretrained_model: Optional[str] = None
    name: Optional[str] = 'Punctuation_and_Capitalization'
    do_training: bool = True
    do_testing: bool = False
    model: PunctuationCapitalizationModelConfig = PunctuationCapitalizationModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='Punctuation_and_Capitalization', files_to_copy=[])


def is_legacy_config(model_cfg: DictConfig) -> bool:
    return 'dataset' in model_cfg or 'class_labels' in model_cfg


def legacy_model_config_to_new_model_config(model_cfg: DictConfig) -> DictConfig:
    train_ds = model_cfg.get('train_ds')
    validation_ds = model_cfg.get('validation_ds')
    test_ds = model_cfg.get('test_ds')
    dataset = model_cfg.dataset
    return OmegaConf.structured(
        PunctuationCapitalizationModelConfig(
            common_dataset_parameters=CommonDatasetParameters(
                pad_label=dataset.pad_label,
                ignore_extra_tokens=dataset.get('ignore_extra_tokens', DEFAULT_IGNORE_EXTRA_TOKENS),
                ignore_start_end=dataset.get('ignore_start_end', DEFAULT_IGNORE_START_END),
                punct_label_ids=model_cfg.punct_label_ids,
                capit_label_ids=model_cfg.capit_label_ids,
            ),
            train_ds=None if train_ds is None else legacy_data_config_to_new_data_config(
                train_ds, dataset, train=True
            ),
            validation_ds=None if validation_ds is None else legacy_data_config_to_new_data_config(
                validation_ds, dataset, train=False
            ),
            test_ds=None if test_ds is None else legacy_data_config_to_new_data_config(test_ds, dataset, train=False),
            punct_head=model_cfg.punct_head,
            capit_head=model_cfg.capit_head,
            tokenizer=model_cfg.tokenizer,
            language_model=model_cfg.language_model,
            optim=model_cfg.optim,
        )
    )
