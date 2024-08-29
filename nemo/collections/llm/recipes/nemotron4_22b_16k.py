from typing import Callable

import torch

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.recipes import nemotron4_22b
from nemo.collections.llm.utils import Partial

NAME = "nemotron4_22b_16k"


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    recipe = nemotron4_22b.pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=fn
    )

    trainer = nemotron4_22b.trainer(
        tensor_parallelism=4,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=5,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = nemotron4_22b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return recipe


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = nemotron4_22b.finetune_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    trainer = nemotron4_22b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=5,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = nemotron4_22b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return recipe
