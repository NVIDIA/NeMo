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

from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
import os
import random
from apex import mpu

from apex.mpu.initialize import (
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
import numpy as np
import torch

from nemo.utils import AppState, logging


def initialize_model_parallel_for_nemo(
    world_size, global_rank, local_rank, tensor_model_parallel_size=1, seed=1234,
):

    # updating NeMo globals
    app_state = AppState()
    app_state.global_rank = global_rank
    app_state.world_size = world_size
    app_state.model_parallel_size = tensor_model_parallel_size
    app_state.model_parallel_rank = compute_model_parallel_rank(local_rank, tensor_model_parallel_size)

    # update apex.mpu globals
    set_tensor_model_parallel_world_size(tensor_model_parallel_size)
    set_tensor_model_parallel_rank(app_state.model_parallel_rank)

    # pipeline model parallelism not implemented in NeMo yet
    set_pipeline_model_parallel_rank(0)
    set_pipeline_model_parallel_world_size(1)

    torch.cuda.set_device(local_rank)

    _set_random_seed(seed)

    app_state._is_megatron_initialized = True


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))
