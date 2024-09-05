from typing import Callable

import torch

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes import nemotron4_22b
from nemo.collections.llm.utils import Config, Partial
from nemo.utils.exp_manager import TimingCallback

NAME = "nemotron4_22b_16k"


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    recipe = nemotron4_22b.pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=fn
    )

    model = nemotron4_22b.model()
    model.config.seq_length = 16384

    trainer = nemotron4_22b.trainer(
        tensor_parallelism=4,
        pipeline_parallelism=1,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=None,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        callbacks=[Config(TimingCallback)],
    )

    data = Config(MockDataModule, seq_length=16384, global_batch_size=512, micro_batch_size=1)

    recipe.model = model
    recipe.trainer = trainer
    recipe.data = data

    return recipe


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = nemotron4_22b.finetune_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    model = nemotron4_22b.model()
    model.config.seq_length = 16384

    trainer = nemotron4_22b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=2,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=None,
        context_parallelism=1,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        callbacks=[Config(TimingCallback)],
    )

    data = Config(SquadDataModule, seq_length=16384, global_batch_size=512, micro_batch_size=1)

    recipe.model = model
    recipe.trainer = trainer
    recipe.data = data

    return recipe
