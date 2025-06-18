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

import os
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging


class CustomTrainingStartCallback(Callback):
    """Custom callback to log a message at the very beginning of training."""

    def __init__(self, model_name: str = "Unknown", model_size: str = "Unknown", custom_prefix: str = "GSW"):
        """
        Initialize the callback with model information.

        Args:
            model_name: Name of the model (e.g., "Nemotron4", "Llama3")
            model_size: Size of the model (e.g., "15B", "70B", "8B")
            custom_prefix: Custom prefix for log messages (default: "GSW")
        """
        self.model_name = model_name
        self.model_size = model_size
        self.custom_prefix = custom_prefix

    def _get_slurm_job_id(self):
        """Extract SLURM job ID from environment variables."""
        # Try different SLURM environment variables
        slurm_vars = ['SLURM_JOB_ID', 'SLURM_JOBID', 'JOB_ID']
        for var in slurm_vars:
            job_id = os.environ.get(var)
            if job_id:
                return job_id
        return "Unknown"

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        full_model_name = f"{self.model_name} {self.model_size}".strip()
        slurm_job_id = self._get_slurm_job_id()

        # TODO: add more info
        # TODO: switch to using our previous format
        # export INFO_STR="GSW: MODEL=${MODEL} FRAMEWORK=nemo2 MODEL_SIZE=${MODEL_SIZE} JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE} DTYPE=${DTYPE} SYNTHETIC_DATA=${SYNTHETIC_DATA_ENABLED^} GSW_VERSION=${GSW_VERSION} FW_VERSION=${FW_VERSION} IMAGE=\'${IMAGE}\' JOB_ID=${SLURM_JOB_ID} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"

        logging.info(
            f"{self.custom_prefix}: MODEL={full_model_name} FRAMEWORK=nemo2 MODEL_SIZE={self.model_size} JOB_NUM_NODES={trainer.num_nodes} GPUS_PER_NODE={trainer.num_devices} DTYPE={trainer.precision} SYNTHETIC_DATA=False GSW_VERSION=0.0.0 FW_VERSION=2.3.0 IMAGE=\'${IMAGE}\' JOB_ID={slurm_job_id} JOB_MODE=training OPTIMIZATION_NAME=\'${OPTIMIZATION_NAME}\' OPTIMIZATION_CODE=\'${OPTIMIZATION_CODE}\' BASE_CONFIG=\'${BASE_CONFIG}\'"
        )

        logging.info(f"{self.custom_prefix}: 🚀 {full_model_name} pre-training initiated!")
        logging.info(f"{self.custom_prefix}: SLURM Job ID: {slurm_job_id}")
        logging.info(
            f"{self.custom_prefix}: Training configuration - Nodes: {trainer.num_nodes}, Devices: {trainer.num_devices}"
        )
        logging.info(f"{self.custom_prefix}: Model: {full_model_name}")
        logging.info(f"{self.custom_prefix}: Precision: {trainer.precision}")

        # Log additional model information if available
        if hasattr(pl_module, 'cfg') and hasattr(pl_module.cfg, 'model'):
            model_cfg = pl_module.cfg.model
            if hasattr(model_cfg, 'num_layers'):
                logging.info(f"{self.custom_prefix}: Number of layers: {model_cfg.num_layers}")
            if hasattr(model_cfg, 'hidden_size'):
                logging.info(f"{self.custom_prefix}: Hidden size: {model_cfg.hidden_size}")
            if hasattr(model_cfg, 'num_attention_heads'):
                logging.info(f"{self.custom_prefix}: Attention heads: {model_cfg.num_attention_heads}")
