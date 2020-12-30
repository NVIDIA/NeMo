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

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.dialogue_state_tracking.sgdqa_model import SGDQAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="sgdqa_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    do_training = True
    if cfg.pretrained_model:
        logging.info(f'Loading pretrained model {cfg.pretrained_model}')
        model = SGDQAModel.from_pretrained(cfg.pretrained_model)
        if do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)
    elif cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path):
        model = SGDQAModel.restore_from(cfg.model.nemo_path)
        logging.info(f'Restoring model from {cfg.model.nemo_path}')
        if do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)
    else:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = SGDQAModel(cfg.model, trainer=trainer)

    if do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.ds_item is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(gpus=gpu)
        model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == '__main__':
    main()
