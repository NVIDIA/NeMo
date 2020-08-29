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
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.common.callbacks import CallbackManager
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Basic run (on CPU for 50 epochs):
    python examples/speaker_recognition/speaker_reco.py \
        model.train_ds.manifest_filepath="<train_manifest_file>" \
        model.validation_ds.manifest_filepath="<validation_manifest_file>" \
        hydra.run.dir="." \
        trainer.gpus=0 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speaker_reco.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

"""

seed_everything(42)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):

    logging.info(f'Hydra config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)

    callbacks = ['LogEpochTimeCallback()', 'LogTrainValidLossCallback()']
    callback_mgr = CallbackManager()
    callbacks = callback_mgr.add_callback(callbacks)
    trainer.callbacks.extend(callbacks)

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(speaker_model)
    model_path = os.path.join(log_dir, '..', 'spkr.nemo')
    speaker_model.save_to(model_path)


if __name__ == '__main__':
    main()
