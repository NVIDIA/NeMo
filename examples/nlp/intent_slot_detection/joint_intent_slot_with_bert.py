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

from nemo.collections.nlp.models.intent_slot_model import IntentSlotModel
from nemo.utils import logging


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.pl.trainer)
    intent_slot_model = IntentSlotModel(cfg.model, trainer=trainer)
    trainer.fit(intent_slot_model)


if __name__ == '__main__':
    main()
