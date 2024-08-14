import torch

from nemo import lightning as nl
from nemo.collections.llm.gpt.trainer.ddp_util import bf16_ddp_config
from nemo.collections.llm.utils import Config, factory


@factory
def llama3_145m_trainer(
    tp_size=1, pp_size=1, pp_dtype=None, vp_size=None, cp_size=1, sp_enable=False, num_nodes=1, callbacks=None
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=bf16_ddp_config,
    )

    trainer = Config(
        nl.Trainer,
        num_nodes=num_nodes,
        devices=8,
        accelerator="gpu",
        # precision="bf16",
        enable_checkpointing=False,
        use_distributed_sampler=False,
        max_steps=1168251,
        log_every_n_steps=10,
        val_check_interval=2000,
        limit_val_batches=32,
        limit_test_batches=50,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        # check_val_every_n_epoch=None,
        strategy=strategy,
        plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        callbacks=callbacks,
    )

    return trainer


@factory
def llama3_8b_trainer(
    tp_size=1, pp_size=1, pp_dtype=None, vp_size=None, cp_size=2, sp_enable=False, num_nodes=1, callbacks=None
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=bf16_ddp_config,
    )

    trainer = Config(
        nl.Trainer,
        num_nodes=num_nodes,
        devices=8,
        accelerator="gpu",
        # precision="bf16",
        enable_checkpointing=False,
        use_distributed_sampler=False,
        max_steps=1168251,
        log_every_n_steps=10,
        val_check_interval=2000,
        limit_val_batches=32,
        limit_test_batches=50,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        # check_val_every_n_epoch=None,
        strategy=strategy,
        plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        callbacks=callbacks,
    )

    return trainer


@factory
def llama3_8b_16k_trainer(
    tp_size=2, pp_size=4, pp_dtype=torch.bfloat16, vp_size=5, cp_size=2, sp_enable=True, num_nodes=2, callbacks=None
) -> Config[nl.Trainer]:
    return llama3_8b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


@factory
def llama3_8b_64k_trainer(
    tp_size=2, pp_size=4, pp_dtype=torch.bfloat16, vp_size=5, cp_size=4, sp_enable=True, num_nodes=4, callbacks=None
) -> Config[nl.Trainer]:
    return llama3_8b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


@factory
def llama3_70b_trainer(
    tp_size=4, pp_size=4, pp_dtype=torch.bfloat16, vp_size=5, cp_size=2, sp_enable=True, num_nodes=8, callbacks=None
) -> Config[nl.Trainer]:
    """LLama3 trainer

    Based on: https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/main/launcher_scripts/conf/training/llama/llama3_70b.yaml#L175-L235


    Returns:
        nl.Trainer: Used for LLama3 pre-training
    """

    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=bf16_ddp_config,
    )

    return Config(
        nl.Trainer,
        devices=8,
        num_nodes=num_nodes,
        accelerator="gpu",
        strategy=strategy,
        plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        max_steps=300000,  # consumed_samples = global_step * global_batch_size
        log_every_n_steps=10,
        val_check_interval=2000,
        limit_val_batches=32,
        limit_test_batches=50,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )


@factory
def llama3_70b_16k_trainer(
    tp_size=8, pp_size=4, pp_dtype=torch.bfloat16, vp_size=5, cp_size=4, sp_enable=True, num_nodes=16, callbacks=None
) -> Config[nl.Trainer]:
    return llama3_70b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


@factory
def llama3_70b_64k_trainer(
    tp_size=8, pp_size=8, pp_dtype=torch.bfloat16, vp_size=5, cp_size=8, sp_enable=True, num_nodes=64, callbacks=None
) -> Config[nl.Trainer]:
    return llama3_70b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )
