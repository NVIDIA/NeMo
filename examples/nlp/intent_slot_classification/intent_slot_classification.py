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

from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="intent_slot_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = IntentSlotClassificationModel(cfg.model, trainer=trainer)
    trainer.fit(model)
    logging.info('Training finished!')
    logging.info("================================================================================================")

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
        logging.info(f'The model is saved into the `.nemo` file: {cfg.model.nemo_path}')

    # after model training is done, if you have saved the checkpoints, you can create the model from
    # the checkpoint again and evaluate it on a data file.
    logging.info("================================================================================================")
    logging.info("Starting the testing of the trained model on test set...")
    logging.info("We will load the latest model saved checkpoint from the training...")

    # retrieve the path to the last checkpoint of the training
    # the latest checkpoint would be used, change to 'best.ckpt' to use the best one instead
    checkpoint_path = os.path.join(
        trainer.checkpoint_callback.dirpath, trainer.checkpoint_callback.prefix + "end.ckpt"
    )
    eval_model = IntentSlotClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # we will setup testing data reusing the same config (test section)
    eval_model.setup_test_data(test_data_config=cfg.model.test_ds)

    trainer.test(model=model, ckpt_path=None, verbose=False)
    logging.info("Testing finished!")
    logging.info("================================================================================================")


if __name__ == '__main__':
    main()
