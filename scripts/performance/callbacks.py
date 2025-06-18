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


class LLMBTrainingStartCallback(Callback):
    """Custom callback to log a message at the very beginning of training."""

    def __init__(
        self,
        model_name: str = "Unknown",
        model_size: str = "Unknown",
        custom_prefix: str = "GSW",
        container_image: str = "Unknown",
    ):
        """
        Initialize the callback with model information.

        Args:
            model_name: Name of the model (e.g., "Nemotron4", "Llama3")
            model_size: Size of the model (e.g., "15B", "70B", "8B")
            custom_prefix: Custom prefix for log messages (default: "GSW")
            container_image: Container image path (e.g., "nvcr.io/nvidia/nemo:2.3.0")
        """
        self.model_name = model_name
        self.model_size = model_size
        self.custom_prefix = custom_prefix
        self.container_image = container_image

    def _get_slurm_job_id(self):
        """Extract SLURM job ID from environment variables."""
        # Try different SLURM environment variables
        slurm_vars = ['SLURM_JOB_ID', 'SLURM_JOBID', 'JOB_ID']
        for var in slurm_vars:
            job_id = os.environ.get(var)
            if job_id:
                return job_id
        return "Unknown"

    def _get_precision(self):
        """Extract DTYPE from environment variables."""
        result = os.environ.get('DTYPE')
        if result:
            return result
        return "Unknown"

    def _get_workload_type(self):
        """Extract WORKLOAD_TYPE from environment variables."""
        result = os.environ.get('WORKLOAD_TYPE')
        if result:
            return result
        return "Unknown"

    def _get_gsw_version(self):
        """Extract GSW_VERSION from environment variables."""
        result = os.environ.get('GSW_VERSION')
        if result:
            return result
        return "Unknown"

    def _get_fw_version(self):
        """Extract FW_VERSION from environment variables."""
        result = os.environ.get('FW_VERSION')
        if result:
            return result
        return "Unknown"

    # must pull the name from environment variable MODEL_NAME because self.model_name is abbreviated for compatibility with other callbacks (e.g., FLOPsMeasurementCallback which is relying on a different convention)
    def _get_model_name(self):
        """Extract MODEL_NAME from environment variables."""
        result = os.environ.get('MODEL_NAME')
        if result:
            return result
        return "Unknown"

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        slurm_job_id = self._get_slurm_job_id()

        logging.info(
            f"{self.custom_prefix}: MODEL={self._get_model_name} FRAMEWORK=nemo2 MODEL_SIZE={self.model_size} JOB_NUM_NODES={trainer.num_nodes} GPUS_PER_NODE={trainer.num_devices} DTYPE={self._get_precision()} SYNTHETIC_DATA=True GSW_VERSION={self._get_gsw_version()} FW_VERSION={self._get_fw_version()} IMAGE='{self.container_image}' JOB_ID={slurm_job_id} JOB_MODE={self._get_workload_type()} OPTIMIZATION_NAME='' OPTIMIZATION_CODE='' BASE_CONFIG=''"
        )

        # Log additional model information if available
        if hasattr(pl_module, 'cfg') and hasattr(pl_module.cfg, 'model'):
            model_cfg = pl_module.cfg.model
            if hasattr(model_cfg, 'num_layers'):
                logging.info(f"{self.custom_prefix}: Number of layers: {model_cfg.num_layers}")
            if hasattr(model_cfg, 'hidden_size'):
                logging.info(f"{self.custom_prefix}: Hidden size: {model_cfg.hidden_size}")
            if hasattr(model_cfg, 'num_attention_heads'):
                logging.info(f"{self.custom_prefix}: Attention heads: {model_cfg.num_attention_heads}")
