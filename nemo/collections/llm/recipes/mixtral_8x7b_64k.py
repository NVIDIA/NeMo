import pytorch_lightning as pl

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.api import squad
from nemo.collections.llm.gpt.model.llama import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.peft.api import gpt_lora
from nemo.collections.llm.recipes import mixtral_8x7b
from nemo.collections.llm.recipes.log.default import default_log
from nemo.collections.llm.recipes.optim.adam import adam_with_cosine_annealing
from nemo.collections.llm.utils import Partial, factory

NAME = "mixtral_8x7b_64k"


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    recipe = mixtral_8x7b.pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=fn
    )

    trainer = mixtral_8x7b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=4,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_parallelism=5,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = mixtral_8x7b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return trainer


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = mixtral_8x7b.finetune_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    trainer = mixtral_8x7b.trainer(
        tensor_parallelism=2,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=5,
        context_parallelism=2,
        sequence_parallelism=True,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    model = mixtral_8x7b.model()
    model.config.seq_length = 16384

    recipe.model = model
    recipe.trainer = trainer

    return trainer
