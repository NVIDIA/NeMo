import nemo_sdk as sdk
from nemo import lightning as nl

## NeMotron 8B


@sdk.factory
def nemotron_8b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


@sdk.factory
def nemotron_8b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


## NeMotron 15B


@sdk.factory
def nemotron_15b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


@sdk.factory
def nemotron_15b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


# NeMotron 22B


@sdk.factory
def nemotron_22b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=10,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


@sdk.factory
def nemotron_22b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=10,
        context_parallel_size=4,
        sequence_parallel=True,
        expert_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )
