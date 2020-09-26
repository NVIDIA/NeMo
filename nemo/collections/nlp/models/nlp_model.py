
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
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from nemo.core.classes import ModelPT
from nemo.utils import logging, AppState

from megatron import mpu

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

__all__ = ['NLPModel']

class NLPModel(ModelPT, ABC):


    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True) -> None:
        """ Override LightningModule DDP initialization if using model parallel"""
        app_state = AppState()

        # we are able initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            LightningModule.init_ddp_connection(self, global_rank, world_size, is_slurm_managing_tasks)
            if app_state.model_parallel_group is None:
                mpu.initialize_model_parallel(app_state.model_parallel_size)
                app_state.model_parallel_group = mpu.get_model_parallel_group()
                app_state.data_parallel_group = mpu.get_data_parallel_group()
                app_state.model_parallel_rank = torch.distributed.get_rank(
                    group=app_state.model_parallel_group
                )
                app_state.data_parallel_rank = torch.distributed.get_rank(
                    group=app_state.data_parallel_group
                )
                logging.info(f'mp_rank: {app_state.model_parallel_rank}')
                logging.info(f'dp_rank: {app_state.data_parallel_rank}')

        else:
            return LightningModule.init_ddp_connection(self, global_rank, world_size, is_slurm_managing_tasks)

    def configure_ddp(self, model, device_ids):
        """ Override LightningModule ddp if using model parallel. """

        logging.info(f'device_ids: {device_ids}')

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info("Configuring DDP for model parallelism.")
            logging.info(f"data_parallel_group: {app_state.data_parallel_group}")
            # with model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial

            model = LightningDistributedDataParallel(
                model,
                device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group
            )
            return model

        else:
            logging.info("Did not detect model parallel using LightningModule.configure_ddp")
            return LightningModule.configure_ddp(self, model, device_ids)
    
    def setup(self, stage):
        """ PTL hook that is called after DDP is initialized """

        if stage == 'fit':

            app_state = AppState()

            if app_state.model_parallel_size is not None:
                logging.info("replacing sampler with model parallel sampler")
                mp_sampler = torch.utils.data.distributed.DistributedSampler(
                    self._train_dl.dataset,
                    num_replicas=app_state.model_parallel_size,
                    rank=app_state.data_parallel_rank
                )
                mp_dl = self._trainer.replace_sampler(self._train_dl, mp_sampler)
                self._train_dl = mp_dl

                if self.bert_model._lazy_init_fn is not None:
                    logging.info(f'Finishing megatron mpu init.')
                    self.bert_model._lazy_init_fn()
                    self._lazy_init_fn = None
                    # model parallel checkpoints need to be restored after torch.distributed is initialized
                    self.bert_model.restore_weights(self.bert_model._restore_path)

