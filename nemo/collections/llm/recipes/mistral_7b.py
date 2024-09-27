import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.api import squad
from nemo.collections.llm.gpt.model.mistral import MistralConfig7B, MistralModel
from nemo.collections.llm.peft.api import gpt_lora
from nemo.collections.llm.recipes.log.default import default_log
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.utils import Config, Partial, factory

NAME = "mistral_7b"


@factory(name=NAME)
def model() -> pl.LightningModule:
    return MistralModel(MistralConfig7B())


@factory(name=NAME)
def trainer(devices=8) -> nl.Trainer:
    """
    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[Config[Callback]]]): List of callback configurations.

    Returns:
        Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=mistral ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)
    """
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        ),
    )

    trainer = Config(
        nl.Trainer,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )


@factory(name=NAME + "_hf")
def hf_resume() -> nl.AutoResume:
    return nl.AutoResume(restore_config=nl.RestoreConfig(path="hf://mistralai/Mistral-7B-v0.3"))


@factory(name=NAME, for_task="llm.pretrain")
def pretrain_recipe() -> Partial:
    return Partial(
        pretrain,
        model=model,
        trainer=trainer,
        data=squad,
        log=default_log,
        optim=distributed_fused_adam_with_cosine_annealing(),
    )


@factory(name=NAME, for_task="llm.finetune")
def finetune_recipe() -> Partial:
    return Partial(
        finetune,
        model=model,
        trainer=trainer,
        data=squad,
        log=default_log,
        optim=distributed_fused_adam_with_cosine_annealing(),
        peft=gpt_lora,
        resume=hf_resume,
    )
