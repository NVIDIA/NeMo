# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from pytorch_lightning.callbacks.callback import Callback

from nemo.utils import logging


class WorkloadInspectorCallback(Callback):
    """
    A Workload Inspector Lightening callback that leverages NVIDIA Nsight Systems (Nsys) profiling.


    This callback enables profiling of specific steps during training using NVIDIA Nsys.
    It allows for precise control over when profiling starts and ends, and provides GPU-wise kernel and exposed comm summaries.

    More info about nsys can be found [here](https://developer.nvidia.com/nsight-systems).

    Args:
        start_step (int): Global batch to start profiling
        end_step (int): Global batch to end profiling

    Example:
        >>> callback = WorkloadInspectorCallback(start_step=100, end_step=200)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
    ):
        assert type(start_step) is int, f'trace capture start_step must be of type int. Found: {type(start_step)}'
        self._wit_profile_start_step = start_step

        assert type(end_step) is int, f'trace capture end_step must be of type int. Found: {type(start_step)}'
        self._wit_profile_end_step = end_step

        assert (
            self._wit_profile_end_step >= self._wit_profile_start_step
        ), 'Tracing end_step must be greater than or equal to start_step'

        logging.info(
            f'profiling setup with start_step: {self._wit_profile_start_step},'
            f'and end_step: {self._wit_profile_end_step}'
        )
        import os
        from workload_inspector.bkg_runner import BackgroundRunner
        from workload_inspector.torch.nsys_downstream import NsysDownstream

        nsys_bg_thread = NsysDownstream(
            os.getenv('NSYS_LOG_DIR', None), os.getenv('GPU_KERN_STATS_OUTPUT_DIR', None), "stdev"
        )
        self.bg_runner = BackgroundRunner(nsys_bg_thread)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> Optional[int]:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
        We use it here to enable nsys profiling via WIT tool on all ranks.
        """

        device = trainer.strategy.root_device
        current_step = trainer.strategy.current_epoch_step
        if device.type == 'cuda':
            if current_step == self._wit_profile_start_step:
                logging.info("====== Start nsys profiling ======")
                torch.cuda.cudart().cudaProfilerStart()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
        We use it here to enable nsys profiling.
        """

        device = trainer.strategy.root_device
        current_step = trainer.strategy.current_epoch_step
        if device.type == 'cuda':
            if current_step == self._wit_profile_end_step:
                logging.info("====== End nsys profiling ======")
                torch.cuda.cudart().cudaProfilerStop()
                self.bg_runner.start_background_task(args=None)
                self.bg_runner.join_background_task()
