from torch.nn import Module
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from typing import (
    Optional,
)


def setup_activation_checkpointing(
    module: Module,
    activation_checkpointing_policy: Optional["_POLICY"] = None,
) -> None:
    if not activation_checkpointing_policy:
        return
    activation_checkpointing_kwargs = {}

    policy = ModuleWrapPolicy(activation_checkpointing_policy)
    activation_checkpointing_kwargs["auto_wrap_policy"] = policy

    if any(isinstance(mod, CheckpointWrapper) for mod in module.modules()):
        rank_zero_warn(
            "FSDP checkpointing is configured, but the model already contains checkpointed layers."
            " Checkpointing will be ignored."
        )
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        **activation_checkpointing_kwargs,
    )
