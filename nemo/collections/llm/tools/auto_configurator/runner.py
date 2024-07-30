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

from typing import List, Optional

from nemo.collections.llm.tools.auto_configurator.autoconfig.search_config import search_configs
from nemo.utils import logging

SUPPORTED_MODELS = [
    "gpt3",
    "llama",
    "mixtral",
    "mistral",
]


class AutoConfigurator:
    """Auto Configurator runner config class."""

    def __init__(
        self,
        model_type: str = None,
        num_nodes: int = None,
        data_paths: List = None,
        tokenizer_path: Optional[str] = None,
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
        max_training_days: Optional[int] = 2,
        max_steps_per_run: Optional[int] = 50,
        vocab_size: Optional[int] = 51200,
        model_args: Optional[dict] = {},
        custom_model: Optional[bool] = False,
        nemo_sdk: Optional[bool] = False,
    ):
        """
        :param str model_type: model type to be used for training.
        :param int num_nodes: number of nodes to be used for training.
        :param List data_paths: list of datafiles to be used for training.
        :param Optional[str] tokenizer_path: path to the tokenizer model.
        :param Optional[int] model_size: size of model to be trained.
        :param Optional[int] model_version: version of model. 3 for GPT3, 2 for Llama2.
        :param Optional[int] gpus_per_node: number of GPUs per node to be used.
        :param Optional[int] gpu_memory_gb: memory per GPU, in GB. Currently 40GB and 80GB A100s/H100s supported.
        :param Optional[str] model_measure: "M" if model_size is specified in millions. "B" if in billions.
        :param Optional[int] seq_length: model sequence length. Available seq_length list for GPT-based models: [2048, 4096, 8192, 16384, 32768].
        :param Optional[int] global_batch_size: model global batch size. Set to "auto" if you want auto configurator to find optimal gbs.
        :param Optional[List[int]] tensor_parallel_sizes: set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
        :param Optional[List[int]] pipeline_parallel_sizes: set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
        :param Optional[List[int]] micro_batch_sizes: set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
        :param Optional[List[int]] context_parallel_sizes: model context parallel size. A list, such as [1, 2, 4, 8].
        :param Optional[List[int]] expert_parallel_sizes: model expert parallel size. A list, such as [1, 2, 4, 8].
        :param Optional[int] min_model_parallel_size: set to "auto" to use our recommendation, or a value for the minimum desired parallelism.
        :param Optional[int] max_model_parallel_size: set to "auto" to use our recommendation, or a value for the maximum desired parallelism.
        :param Optional[int] num_tokens_in_b: number of tokens in billions in train dataset.
        :param Optional[int] tflops_per_gpu: estimated tflops per GPU.
        :param Optional[int] max_minutes_per_run: maximum number of minutes per run for the grid search.
        :param Optional[int] max_training_days: number of days expected model to be trained.
        :param Optional[int] max_steps_per_run: maximum number of steps per run for the grid search.
        :param Optional[int] vocab_size: size of tokenizer vocabulary.
        :param Optional[dict] model_args: additional args to add to mdoel config.
        :param Optional[bool] custom_model: set to True if you want to use custom model.
        :param Optional[bool] nemo_sdk: set to True if you want to run Auto Configurator with nemo-sdk.
        """

        assert model_type in SUPPORTED_MODELS, f"model_type must be set to one of {SUPPORTED_MODELS}."
        assert num_nodes, "num_nodes value must be specified."
        assert data_paths, "training data must be specified."

        self.config = locals()
        self.config.pop('self')

        # Print the config
        logging.info(self._get_message(self.config))

    def generate_configs(self) -> dict:
        """
        :return: dictionary of generated configs.
            key: model config name, type: str.
            value: model config values, type: dict.
        :rtype: dict.
        """

        configs = search_configs(self.config)

        return configs

    def _get_message(self, config) -> str:
        """
        Function that returns runner config line by line.
        :param: dict config: runner config.
        :return: runner config params.
        :rtype: str.
        """

        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message
