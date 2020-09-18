
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

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

__all__ = ['NLPModel']


class NLPModel(ModelPT, ABC):

    def configure_ddp(self, model, device_ids):
        """ Override LightingModule ddp if using model parallel. """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info("Configuring model parallel DDP.")
            # with model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
<<<<<<< HEAD

            # in PTL device_id is trainer.root_gpu
            # TODO: add device_id/root_gpu to AppState?
            device_id = self._trainer.root_gpu
            model = LightningDistributedDataParallel(
                model,
                device_ids=[device_id],
                output_device=device_id,
=======
            # TODO: get device_id
            i = app_state.device_id
            model = LightningDistributedDataParallel(
                model,
                device_ids=[i],
                output_device=i,
>>>>>>> added nlp base model
                process_group=app_state.get_data_parallel_group()
            )
            return model

        else:
            logging.info("Did not detect model parallel using LightningModule.configure_ddp")
            return LightningModule.configure_ddp(self, model, device_ids)

