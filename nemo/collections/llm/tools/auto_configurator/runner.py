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
from nemo.collections.llm.tools.auto_configurator.core.search_config import search_configs
from nemo.collections.llm.utils import Config, Partial
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback

SUPPORTED_MODELS = [
    "gpt3",
    "llama",
    "mixtral",
    "mistral",
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
        model_type: str = None,
        num_nodes: int = None,
        data_paths: List = None,
        path_to_logs: Optional[str] = None,
        tokenizer_type: Optional[str] = "autotokenizer",
        tokenizer_path: Optional[str] = "GPT2BPETokenizer",
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
        nemo_run: Optional[bool] = False,
    ):
        """
        :param str model_type: model type to be used for training.
        :param int num_nodes: number of nodes to be used for training.
        :param List data_paths: list of datafiles to be used for training.
        :param str path_to_logs: path to the directory where the logs will be stored.
        :param Optional[str] tokenizer_type: tokenizer type.
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
        assert tokenizer_type in SUPPORTED_TOKENIZERS, f"tokenizer_type must be set to one of {SUPPORTED_TOKENIZERS}."
        assert num_nodes, "num_nodes value must be specified."
        assert data_paths, "training data must be specified."
        if nemo_run:
            assert path_to_logs, f"path_to_logs parameter must be specified."

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
        if self.config["nemo_run"]:
            configs = self._generate_nemo_run_configs(
                configs,
                self.config["tokenizer_type"],
                self.config["tokenizer_path"],
                self.config["path_to_logs"],
            )

        return configs

    def _generate_nemo_run_configs(
        self,
        configs: dict,
        tokenizer_type: str,
        tokenizer_path: str,
        path_to_logs: str,
    ) -> dict:
        """
        Function that returns a dictionary of Partial configs.
        :param: dict config: runner config.
        :param: str tokenizer_type: tokenizer type.
        :param: str tokenizer_path: path to the tokenizer.
        :param: str path_to_logs: path to logs directory.
        :return: dictionary of Partial configs.
        :rtype: dict.
        """

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

    def _get_model(self, model_config, tokenzier):
        return GPTModel(model_config, tokenizer=tokenizer)
        
    def _get_tokenizer(self, tokenizer_type: str, tokenizer_path: str) -> Config:
        """
        Function that returns the tokenizer config.
        :param: str tokenizer_type: tokenizer type.
        :param: str tokenizer_path: path to the tokenizer.
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
        :param: Config tokenizer: tokenizer config.
        :return: data module.
        :rtype: Config.
        """

        return Config(
            PreTrainingDataModule,
            **data_config,
            tokenizer=tokenizer_config,
        )

    def _get_optim(self, optim_config: Config) -> Config:
        """
        Function that returns the optimizer.
        :param: Config optim_config: optimizer config.
        :return: optimizer.
        :rtype: Config.
        """

        sched = Config(
            CosineAnnealingScheduler,
            warmup_steps=10,
            constant_steps=0,
            min_lr=optim_config.min_lr,
        )

        return Config(
            MegatronOptimizerModule,
            config=optim_config,
            lr_scheduler=sched,
        )

    def _get_trainer(self, trainer_config: dict, strategy: Config) -> Config:
        """
        Function that returns the trainer.
        :param: dict trainer_config: trainer config.
        :param: Config strategy: training strategy.
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
        :param: dict auto_config: model parallelism config.
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
        :param: str run_name: name of run.
        :param: str path_to_logs: path to logs directory.
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
        :param: dict config: runner config.
        :return: runner config params.
        :rtype: str.
        """

        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message
