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

from lightning.pytorch.callbacks import Callback
from nemo.utils import logging


class CustomTrainingStartCallback(Callback):
    """Custom callback to log a message at the very beginning of training."""
    
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        logging.info("GSW: 🚀 Nemotron4 15B pre-training initiated!")
        logging.info(f"GSW: Training configuration - Nodes: {trainer.num_nodes}, Devices: {trainer.num_devices}")
        logging.info(f"GSW: Model: Nemotron4 15B")
        logging.info(f"GSW: Precision: {trainer.precision}") 