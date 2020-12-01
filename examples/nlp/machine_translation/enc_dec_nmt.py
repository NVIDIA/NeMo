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
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTTransformerBase
from nemo.core.config.modelPT import NemoConfig
from examples.nlp.machine_translation.transformer_enc_dec_config import DefaultConfig
from typing import Optional

from hydra.utils import instantiate
from omegaconf import MISSING

from nemo.collections.nlp.models.enc_dec_nlp_model import (
    TokenClassifierConfig,
    TokenizerConfig,
    TransformerDecoderConfig,
    TransformerEmbeddingConfig,
    TransformerEncoderConfig,
    TransformerEncoderDefaultConfig,
)
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import (
    MTEncDecModel,
    MTEncDecModelConfig,
    MTOptimConfig,
    MTSchedConfig,
    TranslationDataConfig,
)
from nemo.core.config import hydra_runner
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager


@dataclass
class MTEncDecConfig(NemoConfig):
    trainer: TrainerConfig = TrainerConfig()

    model: MTTransformerBase = MTTransformerBase()

    optim: MTOptimConfig = MTOptimConfig(sched=MTSchedConfig())

    exp_manager: ExpManagerConfig = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="enc_dec", schema=MTEncDecConfig)
def main(cfg: MTEncDecConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = instantiate(cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    mt_model = MTEncDecModel(cfg.model, trainer=trainer)

    trainer.fit(mt_model)


if __name__ == '__main__':
    main()
