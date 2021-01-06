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

from dataclasses import asdict

import pytorch_lightning as pl

from nemo.collections.asr.models import EncDecCTCModel, configs
from nemo.core.config import optimizers, schedulers
from nemo.utils.exp_manager import exp_manager

"""
python speech_to_text_structured_v2.py
"""

# fmt: off
LABELS = [
    " ", "a", "b", "c", "d", "e",
    "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "'",
]

optim_cfg = optimizers.NovogradParams(
    lr=0.01,
    betas=(0.8, 0.5),
    weight_decay=0.001
)

sched_cfg = schedulers.CosineAnnealingParams(
    warmup_steps=None,
    warmup_ratio=None,
    min_lr=0.0,
)
# fmt: on


def main():
    # Generate default asr model config
    builder = configs.QuartzNetConfigBuilder(name='quartznet_15x5')

    # set global values
    builder.set_repeat(5)
    builder.set_labels(LABELS)
    builder.set_optim(cfg=optim_cfg, sched_cfg=sched_cfg)

    cfg = builder.build()

    # Update values
    # MODEL UPDATES
    cfg.name = "Custom QuartzNet15x5"

    # train ds
    cfg.model.train_ds.manifest_filepath = "<path to train dataset>"

    # validation ds
    cfg.model.validation_ds.manifest_filepath = "<path to test dataset>"

    # Trainer config
    cfg.trainer.gpus = 1
    cfg.trainer.max_epochs = 5

    # Exp Manager config
    cfg.exp_manager.name = cfg.name

    # Note usage of asdict
    trainer = pl.Trainer(**asdict(cfg.trainer))
    exp_manager(trainer, asdict(cfg.exp_manager))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
