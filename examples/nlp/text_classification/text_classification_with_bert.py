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


import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from nemo.utils.exp_manager import exp_manager

from nemo.collections.nlp.models.text_classification.text_classification_model import TextClassificationModel
from nemo.utils import logging


@hydra.main(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.pl.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    text_classification_model = TextClassificationModel(cfg.model, trainer=trainer)
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
