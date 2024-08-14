import torch

from nemo import lightning as nl
from nemo.collections.llm.gpt.trainer.ddp_util import bf16_ddp_config
from nemo.collections.llm.utils import Config, factory


def mixtral_8x3b_trainer(
    tp_size=4,
    pp_size=1,
    pp_dtype=None,
    vp_size=None,
    cp_size=2,
    ep_size=1,
    sp_enable=True,
    num_nodes=1,
    callbacks=None,
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
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
        log_every_n_steps=200,
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
def mixtral_8x3b_16k_trainer(
    tp_size=4,
    pp_size=4,
    pp_dtype=torch.bfloat16,
    vp_size=5,
    cp_size=4,
    ep_size=1,
    sp_enable=True,
    num_nodes=16,
    callbacks=None,
) -> Config[nl.Trainer]:
    return mixtral_8x3b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


@factory
def mixtral_8x3b_64k_trainer(
    tp_size=8,
    pp_size=4,
    pp_dtype=torch.bfloat16,
    vp_size=5,
    cp_size=8,
    ep_size=1,
    sp_enable=True,
    num_nodes=64,
    callbacks=None,
) -> Config[nl.Trainer]:
    return mixtral_8x3b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


def mixtral_8x7b_trainer(
    tp_size=8,
    pp_size=1,
    pp_dtype=None,
    vp_size=None,
    cp_size=1,
    ep_size=1,
    sp_enable=True,
    num_nodes=4,
    callbacks=None,
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
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
        log_every_n_steps=200,
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
def mixtral_8x7b_16k_trainer(
    tp_size=4,
    pp_size=4,
    pp_dtype=torch.bfloat16,
    vp_size=8,
    cp_size=2,
    ep_size=8,
    sp_enable=True,
    num_nodes=32,
    callbacks=None,
) -> Config[nl.Trainer]:
    return mixtral_8x7b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


@factory
def mixtral_8x7b_64k_trainer(
    tp_size=4,
    pp_size=4,
    pp_dtype=torch.bfloat16,
    vp_size=5,
    cp_size=8,
    ep_size=8,
    sp_enable=True,
    num_nodes=128,
    callbacks=None,
) -> Config[nl.Trainer]:
    return mixtral_8x7b_trainer(
        tp_size=tp_size,
        pp_size=pp_size,
        pp_dtype=pp_dtype,
        vp_size=vp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        sp_enable=sp_enable,
        num_nodes=num_nodes,
        callbacks=callbacks,
    )


def mixtral_8x22b_trainer(
    tp_size=4,
    pp_size=4,
    pp_dtype=torch.bfloat16,
    vp_size=8,
    cp_size=2,
    ep_size=2,
    sp_enable=True,
    num_nodes=8,
    callbacks=None,
) -> Config[nl.Trainer]:
    strategy = Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        sequence_parallel=sp_enable,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
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
        log_every_n_steps=200,
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
