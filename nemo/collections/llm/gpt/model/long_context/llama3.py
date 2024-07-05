import nemo_sdk as sdk
from nemo import lightning as nl

## Llama3 8B


@sdk.factory
def llama3_8b_16k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
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


@sdk.factory
def llama3_8b_64k_trainer(devices=8, amp_O2=False) -> nl.Trainer:
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
