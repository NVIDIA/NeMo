from megatron.core.distributed import DistributedDataParallelConfig

from nemo.collections.llm.utils import Config, factory


@factory
def bf16_ddp_config() -> Config[DistributedDataParallelConfig]:
    """Temporary explicit definition of DDPConfig until precision is cleaned up in NeMo"""
    return Config(
        DistributedDataParallelConfig,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )
