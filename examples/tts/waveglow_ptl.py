# Copyright 2020 NVIDIA. All Rights Reserved.
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

import argparse
import math
import os
import time
from functools import partial
from typing import Dict, Optional

import torch
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.core.lightning import LightningModule
from ruamel.yaml import YAML
from torch import nn
from torch.nn.functional import pad


from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, waveglow_log_to_tb_func
from nemo.core.classes import ModelPT
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging
from nemo.utils.arguments import add_optimizer_args, add_scheduler_args


def main():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_optimizer_args(parser, optimizer="adam", default_lr=1e-3, default_opt_args={"weight_decay": 1e-6})
    # parser = add_scheduler_args(parser)
    parser.add_argument("--work_dir", default=None, type=str, help="working directory for experiment")
    parser.add_argument("--train_dataset", default=None, type=str, help="working directory for experiment")
    parser.add_argument("--eval_datasets", default=None, type=str, help="working directory for experiment")
    parser.set_defaults(
        gpus=-1,
        num_nodes=1,
        max_epochs=None,
        gradient_clip_val=0,
        log_save_interval=1000,
        row_log_interval=200,
        check_val_every_n_epoch=25,
        distributed_backend="ddp",
        precision=16,
    )
    args = parser.parse_args()
    if args.max_epochs is None:
        raise ValueError("please use max_epochs")  # TODO: make error message better
    tb_logger = pl_loggers.TensorBoardLogger(args.work_dir)
    lr_logger = LearningRateLogger()
    model = WaveglowPTL(args)
    trainer = Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[lr_logger, LogEpochTimeCallback()])
    trainer.fit(model)


if __name__ == '__main__':
    main()
