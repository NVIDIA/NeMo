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

import nemo.collections.asr as nemo_asr
import nemo.collections.tts.jason as nemo_tts_jason
from nemo.collections.tts.jason.helpers.helpers import get_mask_from_lengths, waveglow_log_to_tb_func
from nemo.core.classes import ModelPT
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging
from nemo.utils.arguments import add_optimizer_args, add_scheduler_args


class WaveglowPTL(ModelPT):
    def __init__(self, args):
        super().__init__()
        self.pad_value = -11.42
        self.sigma = 1.0
        self.audio_to_melspec_precessor = nemo_tts_jason.data.processors.FilterbankFeatures(
            sample_rate=22050,
            n_window_size=1024,
            n_window_stride=256,
            normalize=None,
            n_fft=1024,
            preemph=None,
            nfilt=80,
            lowfreq=0,
            highfreq=None,
            log=True,
            log_zero_guard_type="clamp",
            log_zero_guard_value=1e-5,
            dither=0.0,
            pad_to=8,
            frame_splicing=1,
            pad_value=self.pad_value,
            mag_power=1.0,
            stft_conv=True,
        )
        self.waveglow = nemo_tts_jason.waveglow.waveglow.WaveGlow(
            n_mel_channels=80,
            n_flows=12,
            n_group=8,
            n_early_every=4,
            n_early_size=2,
            WN_config={"n_layers": 8, "n_channels": 32, "kernel_size": 3,},
        )

        # # Set up datasets
        self.__train_dl = self.setup_training_data(args.train_dataset)
        self.__val_dl = self.setup_validation_data(args.eval_datasets)

        # After defining all torch.modules, create optimizer and scheduler
        optimizer_params = {
            'optimizer': args.optimizer,
            'lr': args.lr,
            'opt_args': args.opt_args,
        }
        self.setup_optimization(optimizer_params)
        # iters_per_batch = scheduler_args.pop('iters_per_batch')  # 1 for T2
        # iters_per_batch = 1
        # num_gpus = 1  # TODO: undo hardcode
        # num_samples = len(self.__train_dl.dataset)
        # batch_size = self.__train_dl.batch_size
        # max_steps = math.ceil(num_samples / float(batch_size * iters_per_batch * num_gpus)) * args.max_epochs
        # self.__scheduler = CosineAnnealing(self.__optimizer, max_steps=max_steps, min_lr=1e-5)

    def loss(self, z, log_s_list, log_det_W_list):
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))

    def setup_optimization(self, optim_params: Optional[Dict] = None) -> torch.optim.Optimizer:
        self.__optimizer = super().setup_optimization(optim_params)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        audio, audio_len, = batch
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)

        z, log_s_list, log_det_W_list = self.waveglow((spec, audio))
        loss = self.loss(z=z, log_s_list=log_s_list, log_det_W_list=log_det_W_list)

        output = {
            'loss': loss,  # required
            'progress_bar': {'training_loss': loss},  # optional (MUST ALL BE TENSORS)
            'log': {'loss': loss},
        }
        # return a dict
        return output

    def train_dataloader(self):
        return self.__train_dl

    def setup_training_data(self, path):
        dataset = nemo_tts_jason.data.datalayers.AudioDataset(
            manifest_filepath=path, n_segments=16000, min_duration=0.1, max_duration=None, trim=False,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=dataset._collate_fn)

    def configure_optimizers(self):
        # return [self.__optimizer], [self.__scheduler]
        return self.__optimizer

    def validation_step(self, batch, batch_idx):
        audio, audio_len, = batch
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)

        audio_pred = self.waveglow.infer(spec)
        return {
            "audio_pred": audio_pred,
            "mel_target": spec,
            "mel_len": spec_len,
        }

    def validation_epoch_end(self, outputs):
        waveglow_log_to_tb_func(
            self.logger.experiment,
            outputs[0].values(),
            self.global_step,
            tag="eval",
            mel_fb=self.audio_to_melspec_precessor.fb,
        )
        return {}

    def val_dataloader(self):
        return self.__val_dl

    def setup_validation_data(self, path):
        # TODO: Should n_segments be 16k? But it seems to help with memory footprint
        dataset = nemo_tts_jason.data.datalayers.AudioDataset(
            manifest_filepath=path, n_segments=16000, min_duration=0.1, max_duration=None, trim=False,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=dataset._collate_fn)


class LogEpochTimeCallback(pytorch_lightning.callbacks.base.Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


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
