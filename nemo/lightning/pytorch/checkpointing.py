from torch.nn import Module
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from typing import (
    Optional,
)
from nemo.utils import logging
import torch


def has_llama_mlp(module: Module) -> bool:
    try:
        from transformers.models.llama.modeling_llama import LlamaMLP
    except ImportError:
        return False

    for child in module.modules():
        if isinstance(child, LlamaMLP):
            return True
    return False


class LinearWrapPolicy(ModuleWrapPolicy):
    def __init__(self):
        from transformers.models.llama.modeling_llama import LlamaMLP

        super().__init__(module_classes=(torch.nn.Linear, LlamaMLP))


class LLamaMLPWrapPolicy(ModuleWrapPolicy):
    def __init__(self):
        from transformers.models.llama.modeling_llama import LlamaMLP

        super().__init__(module_classes=(LlamaMLP,))


def setup_activation_checkpointing(
    module: Module,
    activation_checkpointing_policy: Optional["_POLICY"] = None,
) -> None:
    if not activation_checkpointing_policy:
        # use default policy
        if has_llama_mlp(module):
            policy = LLamaMLPWrapPolicy()
        else:
            # wrap all linear layers
            policy = LinearWrapPolicy()

    activation_checkpointing_kwargs = {}
    if activation_checkpointing_policy:
        policy = ModuleWrapPolicy(activation_checkpointing_policy)
    activation_checkpointing_kwargs["auto_wrap_policy"] = policy
    if any(isinstance(mod, CheckpointWrapper) for mod in module.modules()):
        logging.warning(
            "The model already contains checkpointed layers."
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
    print(module)
