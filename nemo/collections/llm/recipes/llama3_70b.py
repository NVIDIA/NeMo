from typing import Callable

import pytorch_lightning as pl
import torch

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config70B, LlamaModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, hf_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.trainer.default import default_trainer
from nemo.collections.llm.utils import Config, Partial
from nemo.utils.exp_manager import TimingCallback

NAME = "llama3_70b"


def model() -> Config[pl.LightningModule]:
    return Config(LlamaModel, config=Config(Llama3Config70B))


def pretrain_recipe(
    name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int, fn: Callable = pretrain
) -> Partial:
    return Partial(
        fn,
        model=model(),
        trainer=default_trainer(
            tensor_parallelism=4,
            pipeline_parallelism=4,
            pipeline_parallelism_type=torch.bfloat16,
            virtual_pipeline_parallelism=5,
            context_parallelism=2,
            sequence_parallelism=True,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[Config(TimingCallback)],
            ckpt_async_save=True,
            ckpt_parallel_load=True,
        ),
        data=Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(ckpt_dir=ckpt_dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


def finetune_recipe(name: str, ckpt_dir: str, num_nodes: int, num_gpus_per_node: int) -> Partial:
    recipe = pretrain_recipe(
        name=name, ckpt_dir=ckpt_dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=finetune
    )
    recipe.resume = hf_resume("hf://meta-llama/Meta-Llama-3-70B")
    recipe.peft = Config(LoRA)
    recipe.data = Config(SquadDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1)
    return recipe
