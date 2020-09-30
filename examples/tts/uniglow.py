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

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import UniGlowModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="uniglow")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([epoch_time_logger])
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.resume_from_ckpt is None:
        logging.info("Training UniGlow from scratch")
        model = UniGlowModel(cfg=cfg.model, trainer=trainer)
    else:
        logging.info("Fine-tuning UniGlow from {cfg.resume_from_ckpt}")
        model = UniGlowModel.restore_from(cfg.resume_from_ckpt)
        model.setup_training_data(cfg.model.train_ds)
        model.setup_validation_data(cfg.model.validation_ds)

    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
