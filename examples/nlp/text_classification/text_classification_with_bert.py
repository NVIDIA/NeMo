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


import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Optional, Union

from nemo.collections.nlp.models.text_classification_model import (
    TextClassificationModel,
    TextClassificationModelConfig,
)
from nemo.utils import logging
from nemo.core.config.pytorch_lightning import TrainerConfig


cs = ConfigStore.instance()
cs.store(group="model", name="text_classification_with_bert", node=TextClassificationModelConfig)
cs.store(group="trainer", name="pl", node=TrainerConfig)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.model.trainer = cfg.trainer
    logging.info(f'Config: {cfg.pretty()}')
    text_classification_model = TextClassificationModel(cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
