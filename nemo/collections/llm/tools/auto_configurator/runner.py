# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo.collections.llm.tools.auto_configurator.autoconfig.search_config import search_configs
from typing import List, Optional
from nemo.utils import logging


SUPPORTED_MODELS = [
    "gpt3",
    "llama",
    "mixtral",
    "mistral",
]


class AutoConfigurator():
    def __init__(
        self,
        model_type: str = None,
        num_nodes: int = None,
        model_size: Optional[int] = None,
        model_version: Optional[int] = None,
        gpus_per_node: Optional[int] = 8,
        gpu_memory_gb: Optional[int] = 80,
        model_measure: Optional[str] = "B",
        seq_length: Optional[int] = 2048,
        global_batch_size: Optional[int] = "auto",
        tensor_parallel_sizes: Optional[List[int]] = "auto",
        pipeline_parallel_sizes: Optional[List[int]] = "auto",
        micro_batch_sizes: Optional[List[int]] = "auto",
        context_parallel_sizes: Optional[List[int]] = [1],
        expert_parallel_sizes: Optional[List[int]] = [1],
        min_model_parallel_size: Optional[int] = "auto",
        max_model_parallel_size: Optional[int] = "auto",
        num_tokens_in_b: Optional[int] = 300,
        tflops_per_gpu: Optional[int] = 140,
        max_minutes_per_run: Optional[int] = 30,
        vocab_size: Optional[int] = 51200,
        max_training_days: Optional[int] = 2,
    ):  

        assert model_type in SUPPORTED_MODELS, f"model_type must be set to one of {SUPPORTED_MODELS}."
        assert num_nodes, "num_nodes value must be specified."

        self.config = locals()
        self.config.pop('self')

        # Print the config
        logging.info(self._get_message(self.config))
    
    def get_configs(self):
        configs = search_configs(self.config)

        return configs
    
    def _get_message(self, config):
        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message