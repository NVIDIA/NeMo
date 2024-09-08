from typing import Optional, Union

import torch
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed_plugin
from nemo.collections.llm.utils import Config, Partial


def default_trainer(
    tensor_parallelism: int,
    pipeline_parallelism: int,
    pipeline_parallelism_type: Optional[torch.dtype],
    virtual_pipeline_parallelism: Optional[int],
    context_parallelism: int,
    sequence_parallelism: bool,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[Config[Callback]]] = None,
    limit_test_batches: Optional[Union[int, float]] = 50,
    limit_val_batches: Optional[Union[int, float]] = 32,
    val_check_interval: Optional[Union[int, float]] = 2000,
    ckpt_async_save: bool = False,
    ckpt_parallel_load: bool = False,
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=ckpt_async_save,
        ckpt_parallel_load=ckpt_parallel_load,
    )

    trainer = Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed_plugin(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
    )

    return trainer
