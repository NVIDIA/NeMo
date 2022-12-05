# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# hack to install necessary nemo system-libraries, without having to create a sagemaker compatible container.
# fmt: off
import os; os.system('apt-get -qq update && apt-get -qq install -y libsndfile1 ffmpeg')
# fmt: on
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = ASRModel.from_pretrained(cfg.model_name, map_location='cpu')

    # set new vocabulary if required
    if cfg.labels is not None:
        asr_model.change_vocabulary(cfg.labels)

    if cfg.freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")

    # data & augmentation setup
    asr_model.setup_training_data(cfg.train_ds)
    asr_model.setup_multiple_validation_data(cfg.validation_ds)
    asr_model.spec_augmentation = asr_model.from_config_dict(asr_model.cfg.spec_augment)

    # set metric variables
    asr_model._wer.use_cer = cfg.use_cer
    asr_model._wer.log_prediction = cfg.log_prediction

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
