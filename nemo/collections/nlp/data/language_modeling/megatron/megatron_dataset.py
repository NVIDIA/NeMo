# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron dataset class that handles initialization and other megatron specific things common to all datasets."""

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.utils import AppState, logging


class MegatronDataset(torch.utils.data.Dataset):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        app_state = AppState()

        if not app_state._is_megatron_initialized:
            logging.info(
                f"Initializing megatron since it hasn't been initialized by the model. This is normal if you are using a NeMo model with Megatron dataloaders."
            )
            app_state.global_rank = trainer.global_rank
            app_state.world_size = trainer.world_size
            app_state.model_parallel_size = 1
            app_state.model_parallel_rank = trainer.global_rank

            initialize_model_parallel_for_nemo(
                world_size=trainer.world_size,
                global_rank=trainer.global_rank,
                local_rank=trainer.local_rank,
                tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
                seed=self.cfg.get('seed', 1234),
            )
