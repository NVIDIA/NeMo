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

from typing import List

import torch
from megatron import mpu
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules import MegatronBertEncoder
from nemo.core.classes import ModelPT
from nemo.utils import AppState, app_state, logging

__all__ = ['NLPModel']


class NLPModel(ModelPT):
    """Base class for NLP Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.bert_model = None  # Pretrained BERT encoder
        self.set_world_size(trainer)

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Override for LightningModule DDP initialization.
            Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            mpu.initialize_model_parallel(app_state.model_parallel_size)
            app_state.model_parallel_group = mpu.get_model_parallel_group()
            app_state.data_parallel_group = mpu.get_data_parallel_group()
            app_state.model_parallel_rank = torch.distributed.get_rank(group=app_state.model_parallel_group)
            app_state.data_parallel_rank = torch.distributed.get_rank(group=app_state.data_parallel_group)
            logging.info(f'mp_rank: {app_state.model_parallel_rank}')
            logging.info(f'dp_rank: {app_state.data_parallel_rank}')

    def configure_ddp(self, model: LightningModule, device_ids: List[int]) -> DistributedDataParallel:
        """ Override LightningModule ddp if using model parallel.

        Args:
            model (LightningModule): the LightningModule currently being optimized
            device_ids (List[int]): the list of GPU ids.

        Returns:
            DistributedDataParallel: DDP wrapped model
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info("Configuring DDP for model parallelism.")
            logging.info(f"data_parallel_group: {app_state.data_parallel_group}")
            # with model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial

            model = LightningDistributedDataParallel(
                model, device_ids, output_device=device_ids[0], process_group=app_state.data_parallel_group
            )
            return model

        else:
            logging.info("Did not detect model parallel using LightningModule.configure_ddp")
            return LightningModule.configure_ddp(self, model, device_ids)

    def setup(self, stage: str) -> None:
        """ PTL hook that is called after DDP is initialized.
            Called at the beginning of fit and test. 

        Args:
            stage (str): either 'fit' or 'test'
        """

        # TODO: implement model parallel for test stage
        if stage == 'fit':

            app_state = AppState()

            if app_state.model_parallel_size is not None:

                if app_state.model_parallel_group is None:
                    self.init_model_parallel(app_state.global_rank, app_state.world_size)

                # Update PTL trainer to use our configure_ddp
                self._trainer.accelerator_backend.configure_ddp = self.configure_ddp

                if isinstance(self.bert_model, MegatronBertEncoder):
                    logging.info(f"restoring model parallel checkpoint: {self.bert_model._restore_path}")
                    # model parallel checkpoints need to be restored after torch.distributed is initialized
                    self.bert_model.restore_weights(self.bert_model._restore_path)

                    logging.info("replacing sampler with model parallel sampler")
                    mp_sampler = torch.utils.data.distributed.DistributedSampler(
                        self._train_dl.dataset,
                        num_replicas=app_state.data_parallel_size,
                        rank=app_state.data_parallel_rank,
                    )
                    mp_dl = self._trainer.replace_sampler(self._train_dl, mp_sampler)
                    self._train_dl = mp_dl
                else:
                    raise NotImplementedError(
                        f'The BERT encoder: {self.bert_model} does not support model parallelism yet.'
                    )
