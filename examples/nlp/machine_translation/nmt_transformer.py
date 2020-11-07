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


import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.common.callbacks import MachineTranslationLogEvalCallback
from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="toy_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    transformer_mt = TransformerMTModel(cfg.model, trainer=trainer)
    trainer.fit(transformer_mt)
    transformer_mt.save_to("transformer.nemo")
    translation = transformer_mt.translate(text=['Schulen werden zu größerem Fokus auf Mathematik, Rechtschreibung und Grammatik angehalten'])
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAA')
    print(translation)


if __name__ == '__main__':
    main()
