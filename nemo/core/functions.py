# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from typing import List, Optional

# from nemo.backends.pytorch import pytorch_fit
# from nemo.utils.app_state import AppState
# from nemo.core import Backend
from ..backends.pytorch import pytorch_fit
from ..utils.app_state import AppState
from .neural_factory import Backend


def fit(
    tensors_to_optimize,
    training_graph=None,
    optimizer=None,
    optimization_params=None,
    callbacks: list = None,
    lr_policy=None,
    batches_per_step=None,
    stop_on_nan_loss=False,
    steps_per_nan_check=100,
    synced_batchnorm=False,
    synced_batchnorm_groupsize=0,
    gradient_predivide=False,
    amp_max_loss_scale=2.0 ** 24,
    reset=False,
):
    app_state = AppState()
    if app_state.backend == Backend.PyTorch:
        pytorch_fit(
            tensors_to_optimize,
            training_graph,
            optimizer,
            optimization_params,
            callbacks,
            lr_policy,
            batches_per_step,
            stop_on_nan_loss,
            steps_per_nan_check,
            synced_batchnorm,
            synced_batchnorm_groupsize,
            gradient_predivide,
            amp_max_loss_scale,
            reset,
        )
    else:
        raise NotImplemented(f"Backend: {app_state.backend} is not supported")
