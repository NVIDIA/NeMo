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

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer, SentencePieceTokenizer
from nemo.collections.llm import GPTModel, PreTrainingDataModule
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.tools.auto_configurator.core.training_config import generate_grid_search_configs
from nemo.collections.llm.tools.auto_configurator.core.utils import generic_base_config
from nemo.collections.llm.utils import Config, Partial
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback

SUPPORTED_MODELS = [
    "gpt3",
    "llama",
    "mixtral",
    "mistral",
    "gemma",
    "nemotron",
]

SUPPORTED_TOKENIZERS = [
    "autotokenizer",
    "sentencepiece",
    "huggingface",
]


class AutoConfigurator:
    """Auto Configurator runner config class."""

    def __init__(
        self,
        model: Config = None,
        num_nodes: int = None,
        data_paths: List = None,
        path_to_logs: Optional[str] = None,
        tokenizer_type: Optional[str] = "autotokenizer",
        tokenizer_path: Optional[str] = "GPT2BPETokenizer",
        model_size: Optional[int] = None,
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
    ):
        """
        Args:
            model_type (Config): model type to be used for training.
            num_nodes (int): number of nodes to be used for training.
            data_paths (List): list of datafiles to be used for training.
            path_to_logs (str): path to the directory where the logs will be stored.
            tokenizer_type (Optional[str]): tokenizer type.
            tokenizer_path (Optional[str]): path to the tokenizer model.
            model_size (Optional[int]): size of model to be trained.
            gpus_per_node (Optional[int]): number of GPUs per node to be used.
            gpu_memory_gb (Optional[int]): memory per GPU, in GB. Currently 40GB and 80GB A100s/H100s supported.
            seq_length (Optional[int]): model sequence length. Available seq_length list for GPT-based models: [2048, 4096, 8192, 16384, 32768].
            global_batch_size (Optional[int]): model global batch size. Set to "auto" if you want auto configurator to find optimal gbs.
            tensor_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            pipeline_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            micro_batch_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            context_parallel_sizes (Optional[List[int]]): model context parallel size. A list, such as [1, 2, 4, 8].
            expert_parallel_sizes (Optional[List[int]]): model expert parallel size. A list, such as [1, 2, 4, 8].
            min_model_parallel_size (Optional[int]): set to "auto" to use our recommendation, or a value for the minimum desired parallelism.
            max_model_parallel_size (Optional[int]): set to "auto" to use our recommendation, or a value for the maximum desired parallelism.
            num_tokens_in_b (Optional[int]): number of tokens in billions in train dataset.
            tflops_per_gpu (Optional[int]): estimated tflops per GPU.
            max_minutes_per_run (Optional[int]): maximum number of minutes per run for the grid search.
            max_training_days (Optional[int]): number of days expected model to be trained.
            max_steps_per_run (Optional[int]): maximum number of steps per run for the grid search.
            vocab_size (Optional[int]): size of tokenizer vocabulary.
        """

        model_type = self._get_model_type(model.__class__.__name__)
        assert model_type in SUPPORTED_MODELS, f"model_type must be set to one of {SUPPORTED_MODELS}."
        assert tokenizer_type in SUPPORTED_TOKENIZERS, f"tokenizer_type must be set to one of {SUPPORTED_TOKENIZERS}."
        assert num_nodes, "num_nodes value must be specified."
        assert data_paths, "training data must be specified."
        assert path_to_logs, f"path_to_logs parameter must be specified."
        gpu_count = num_nodes * gpus_per_node
        assert gpu_count > 0, "num_nodes * gpus_per_node must be an int larger than zero."
        assert gpu_memory_gb in (
            40,
            80,
        ), "gpu_memory_gb can only be 40 or 80."
        assert max_minutes_per_run >= 10, "max_minutes_per_run must be an int and be at least 10 minutes."

        self.config = locals()
        self.config.pop('self')
        self.config["model_type"] = model_type
        self.config["model_size_in_b"] = model_size
        self.config["gpu_count"] = gpu_count
        self.config["num_gpus"] = gpus_per_node

        # Print the config
        logging.info(self._get_message(self.config))

    def generate_configs(self) -> dict:
        """
        Function that returns a dictionary of Partial configs.
        : dict config: runner config.
        : str tokenizer_type: tokenizer type.
        : str tokenizer_path: path to the tokenizer.
        : str path_to_logs: path to logs directory.
        :return: dictionary of Partial configs.
        :rtype: dict.
        """

        # Generate base config for the given model size
        base_cfg, train_cfg = generic_base_config(
            model_name=self.config["model_type"],
            model_size_in_b=self.config["model_size"],
            cfg=self.config,
        )

        # Launch grid search for training constraints
        configs = generate_grid_search_configs(base_cfg, train_cfg)

        tokenizer_type = self.config.get("tokenizer_type")
        tokenizer_path = self.config.get("tokenizer_path")
        path_to_logs = self.config.get("path_to_logs")

        tokenizer = self._get_tokenizer(tokenizer_type, tokenizer_path)
        for name, config in configs.items():
            strategy = self._get_startegy(config['auto_config'])
            configs[name] = Partial(
                pretrain,
                model=self._get_model(config['model'], tokenizer),
                trainer=self._get_trainer(config['trainer'], strategy),
                data=self._get_data(config['data'], tokenizer),
                optim=self._get_optim(config['optim']),
                log=self._get_logger(name, path_to_logs),
                resume=None,
            )

        return configs

    def _get_model(self, model_config, tokenizer):
        return GPTModel(model_config, tokenizer=tokenizer)

    def _get_tokenizer(self, tokenizer_type: str, tokenizer_path: str) -> Config:
        """
        Function that returns the tokenizer config.
        : str tokenizer_type: tokenizer type.
        : str tokenizer_path: path to the tokenizer.
        :return: tokenizer config.
        :rtype: Config.
        """

        if tokenizer_type == "sentencepiece":
            return Config(SentencePieceTokenizer, model_path=tokenizer_path)
        else:
            return Config(AutoTokenizer, pretrained_model_name=tokenizer_path)

    def _get_data(self, data_config: dict, tokenizer_config: Config) -> Config:
        """
        Function that returns the data module.
        : Config tokenizer: tokenizer config.
        :return: data module.
        :rtype: Config.
        """

        return Config(
            PreTrainingDataModule,
            **data_config,
            tokenizer=tokenizer_config,
        )

    def _get_trainer(self, trainer_config: dict, strategy: Config) -> Config:
        """
        Function that returns the trainer.
        : dict trainer_config: trainer config.
        : Config strategy: training strategy.
        :return: trainer.
        :rtype: Config.
        """

        return Config(
            nl.Trainer,
            **trainer_config,
            strategy=strategy,
            plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
            callbacks=[Config(TimingCallback)],
        )

    def _get_startegy(self, auto_config: dict) -> Config:
        """
        Function that returns the training strategy.
        : dict auto_config: model parallelism config.
        :return: training strategy.
        :rtype: Config.
        """

        return Config(
            nl.MegatronStrategy,
            pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=auto_config.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=auto_config.get('pipeline_model_parallel_size', 1),
            virtual_pipeline_model_parallel_size=auto_config.get('virtual_pipeline_model_parallel_size', None),
            context_parallel_size=auto_config.get('context_parallel_size', 1),
            expert_model_parallel_size=auto_config.get('expert_model_parallel_size', 1),
        )

    def _get_logger(self, run_name: str, path_to_logs: str) -> Config:
        """
        Function that returns the training strategy.
        : str run_name: name of run.
        : str path_to_logs: path to logs directory.
        :return: training logger.
        :rtype: Config.
        """

        tb_logger = Config(TensorBoardLogger, save_dir=path_to_logs)

        ckpt = Config(
            nl.ModelCheckpoint,
            monitor="reduced_train_loss",
            save_best_model=False,
            save_last=False,
            save_top_k=0,
        )

        return Config(
            nl.NeMoLogger,
            ckpt=ckpt,
            name=run_name,
            tensorboard=tb_logger,
            wandb=None,
            dir=path_to_logs,
        )

    def _get_message(self, config: dict) -> str:
        """
        Function that returns runner config line by line.
        : dict config: runner config.
        :return: runner config params.
        :rtype: str.
        """

        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message

    def _get_model_type(self, model: str) -> str:
        if "GPT" in model:
            return "gpt3"
        elif "Llama" in model:
            return "Llama"
        elif "Mixtral" in model:
            return "mixtral"
        elif "Mistral" in model:
            return "mistral"
        elif "Gemma" in model:
            return "gemma"
        elif "Nemotron" in model:
            return "nemotron"
        else:
            return None
