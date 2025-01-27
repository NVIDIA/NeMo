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


def find_module_by_class_name(model: Module, class_name: str) -> Optional[Module]:
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            return module
    return None


def setup_activation_checkpointing(
    module: Module,
    layer_names: Optional[list[str]] = None,
) -> None:
    if not layer_names:
        return
    module_types = []

    for layer_name in layer_names:
        module_type = type(find_module_by_class_name(module, layer_name))
        if module_type is not None and module_type not in module_types:
            module_types.append(module_type)

    activation_checkpointing_kwargs = {}
    policy = ModuleWrapPolicy(module_types)
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
