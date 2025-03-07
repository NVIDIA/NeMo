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

import torch
from lightning.pytorch.callbacks import Callback


class CrashSimulationCallback(Callback):
    def __init__(self, crash_step=17):
        self.crash_step = crash_step
        self.has_simulated_crash_happened = False
        print(f"Setup to simulate a crash if step == {self.crash_step} and a crash hasn't been simulated before")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.crash_step and trainer.global_step == self.crash_step and not self.has_simulated_crash_happened:
            raise Exception(f"Simulating a crash at step {self.crash_step}!")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if not self.has_simulated_crash_happened:
            self.has_simulated_crash_happened = True
            print(f"Resuming from checkpoint, setting has_simulated_crash_happened to True!")
