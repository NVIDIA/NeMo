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
import time
from typing import List, Union

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

from nemo.utils import logging


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log
    """

    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


class LogTrainValidLossCallback(Callback):
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logging.info("Training started")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        print_freq = trainer.row_log_interval
        total_batches = trainer.num_training_batches
        if 0 < print_freq < 1:
            print_freq = int(total_batches*print_freq)
        if batch_idx % print_freq == 0:
            logging.info(
                "Epoch: {}/{} batch: {}/{} train_loss: {:.3f} train_acc: {:.2f}".format(
                    trainer.current_epoch + 1,
                    trainer.max_epochs,
                    batch_idx + 1,
                    total_batches,
                    pl_module.loss_value,
                    pl_module.accuracy,
                )
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        logging.info(
            "----> Epoch: {}/{} val_loss: {:.3f} val_acc: {:.2f} <----".format(
                trainer.current_epoch + 1, trainer.max_epochs, pl_module.val_loss_mean, pl_module.accuracy
            )
        )
