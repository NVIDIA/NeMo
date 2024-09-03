import nemo_run as run
import torch

from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision


@run.cli.factory
def bf16_mixed() -> run.Config[MegatronMixedPrecision]:
    return run.Config(
        MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


@run.cli.factory
def fp16_mixed() -> run.Config[MegatronMixedPrecision]:
    return run.Config(
        MegatronMixedPrecision,
        precision="16-mixed",
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
