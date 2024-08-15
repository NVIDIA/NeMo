import torch
from megatron.core.optimizer import OptimizerConfig

from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule, OptimizerModule


def distributed_fused_adam_with_cosine_annealing() -> Config[OptimizerModule]:
    opt_cfg = Config(
        OptimizerConfig,
        optimizer="adam",
        lr=0.0001,
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
    )

    sched = Config(
        CosineAnnealingScheduler,
        warmup_steps=500,
        constant_steps=0,
        min_lr=1.0e-05,
    )

    return Config(
        MegatronOptimizerModule,
        config=opt_cfg,
        lr_scheduler=sched,
    )
