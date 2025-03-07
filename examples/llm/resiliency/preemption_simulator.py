# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import signal

import torch
from lightning.pytorch.callbacks import Callback


class PreemptionSimulationCallback(Callback):
    def __init__(self, preemption_step=4):
        self.preemption_step = preemption_step
        print(f"Setup to simulate a preemption if step == {self.preemption_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.preemption_step and trainer.global_step == self.preemption_step:
            print(f"Simulating preemption by raising a SIGTERM at step {self.preemption_step}!")
            signal.raise_signal(signal.SIGTERM)
