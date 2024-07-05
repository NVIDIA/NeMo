import nemo_sdk as sdk
from nemo import lightning as nl

# Mixtral 8B


@sdk.factory
def mixtral_8x3b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=8,
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
def mixtral_8x3b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=8,
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


# Mixtral 8x7B


@sdk.factory
def mixtral_8x7b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_model_parallel_size=4,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=amp_O2),
    )


@sdk.factory
def mixtral_8x7b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=8,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=8,
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
