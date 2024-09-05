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
import pytorch_lightning as pl
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.llm.api import pretrain, finetune
from nemo.collections.llm.recipes import llama3_70b
import nemo_run as run
import pytorch_lightning as pl
import torch


NAME = "llama3_70b_64k"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    model_config = llama3_70b.model()
    model_config.config.seq_length = 65536
    return model_config


def trainer(
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Config:
    return llama3_70b.trainer(
        tensor_parallelism=8,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=5,
        context_parallelism=8,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        callbacks=[run.Config(TimingCallback)],
    )


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = llama3_70b.pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    recipe.data = run.Config(MockDataModule, seq_length=65536, global_batch_size=512, micro_batch_size=1)

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = llama3_70b.finetune_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    recipe.data = run.Config(SquadDataModule, seq_length=65536, global_batch_size=512, micro_batch_size=1)

    return recipe
