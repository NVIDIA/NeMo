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

from dataclasses import dataclass, field

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer, SentencePieceTokenizer
from nemo.collections.llm import GPTModel, PreTrainingDataModule
from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


@dataclass
class ModelConfig:
    def __init__(
        self,
        config=None,
    ):
        """
        Args:
            name (str): model name.
            version (int): model version.
            size (int):  model size.
            measure (str): meausre of model size. "M" if model size in millions, "B" if in billions.
            cfg (dict): auto configurator runner config.
        """

        self.config = config

    def get_model(self):
        """Function that returns model config."""

        self.config.model.global_batch_size = self.config.global_batch_size
        self.config.model.seq_length = self.config.seq_length

        return self.config.model

    def get_optim(self) -> OptimizerConfig:
        """Function that returns optimizer config.

        Returns:
            OptimizerConfig: optimizer config.
        """
        optim_params = {
            "optimizer": "adam",
            "lr": 1e-4,
            "min_lr": 1e-5,
            "use_distributed_optimizer": True,
            "bf16": True,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "overlap_grad_reduce": False,
            "overlap_param_gather": True,
        }

        optim_config = Config(
            OptimizerConfig,
            **optim_params,
        )

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

    def get_trainer(self) -> dict:
        """Function that returns config for PTL trainer.

        Returns:
            Config: trainer config.
        """

        trainer_config = {
            "accelerator": "gpu",
            "enable_checkpointing": False,
            "use_distributed_sampler": False,
            "max_epochs": None,
            "log_every_n_steps": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "num_nodes": self.config.num_nodes,
            "devices": self.config.num_gpus,
            "max_steps": self.config.max_steps_per_run,
            "val_check_interval": self.config.max_steps_per_run,
        }

        strategy = Config(
            nl.MegatronStrategy,
            pipeline_dtype=torch.bfloat16,
        )

        return Config(
            nl.Trainer,
            **trainer_config,
            strategy=strategy,
            plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
            callbacks=[Config(TimingCallback)],
        )

        return trainer_config

    def get_tokenizer(self, tokenizer_type: str, tokenizer_path: str) -> Config:
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

    def get_data(self) -> dict:
        """Function that returns dataset config.

        Returns:
            dict: data config.
        """

        # Data config
        data_config = {
            "paths": self.config.data_paths,
            "seq_length": self.config.seq_length,
            "global_batch_size": self.config.global_batch_size,
            "num_workers": 2,
            "index_mapping_dir": None,
        }

        # Define the tokenizer
        tokenizer = self.get_tokenizer(
            self.config.tokenizer_type,
            self.config.tokenizer_path,
        )

        return Config(
            PreTrainingDataModule,
            **data_config,
            tokenizer=tokenizer,
        )

    def get_logger(self) -> Config:
        """
        Function that returns the training strategy.
        : str run_name: name of run.
        : str path_to_logs: path to logs directory.
        :return: training logger.
        :rtype: Config.
        """

        # Define TensorBoard Logger
        tb_logger = Config(TensorBoardLogger, save_dir=self.config.path_to_logs)

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
            tensorboard=tb_logger,
            wandb=None,
            dir=self.config.path_to_logs,
        )

    def get_run_config(self) -> dict:
        """Function that returns config for cluster job.

        Returns:
            dict: cluster job config.
        """

        run_config = {
            "name": self.config.model.__class__.__name__,
            "results_dir": None,
            "time_limit": f"0-00:{self.config.max_minutes_per_run}:00",
        }

        return run_config
