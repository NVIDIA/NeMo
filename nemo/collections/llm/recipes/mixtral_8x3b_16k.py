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


from typing import Optional

import torch
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
import pytorch_lightning as pl
from nemo.collections.llm.api import pretrain, finetune
from nemo.collections.llm.recipes import mixtral_8x3b
import nemo_run as run
import pytorch_lightning as pl
import torch


NAME = "mixtral_8x3b_16k"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    model_config = mixtral_8x3b.model()
    model_config.config.seq_length = 16384
    model_config.config.max_position_embeddings = 16384
    return model_config


def trainer(
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Config:
    return mixtral_8x3b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=2,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=8,
        context_parallelism=2,
        sequence_parallelism=True,
        expert_parallelism=1,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = mixtral_8x3b.pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    recipe.data = run.Config(MockDataModule, seq_length=16384, global_batch_size=512, micro_batch_size=1)

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = mixtral_8x3b.finetune_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    recipe.data = run.Config(SquadDataModule, seq_length=16384, global_batch_size=512, micro_batch_size=1)

    return recipe
