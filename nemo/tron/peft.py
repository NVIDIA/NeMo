# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from nemo.collections.llm.fn.base import walk

class PEFT(ABC):
    """Abstract base class for Parameter-Efficient Fine-Tuning (PEFT) methods.

    This class defines the minimal interface for PEFT methods, manages the
    application of the transform to a base model, and freezes base parameters.

    Subclasses are responsible for managing their own configuration (e.g., rank,
    target modules) and implementing the specific `transform` logic. Identifying
    which parameters belong to the adapter is also delegated to subclasses via
    `set_params_to_save`.
    """

    def __init__(self, training: bool = True):
        """Initializes the base PEFT handler."""
        super().__init__()
        self.training = training
        self.params_to_save: set[str] = set()


    @abstractmethod
    def transform(
        self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
    ) -> nn.Module:
        """Transforms a single module according to the PEFT method.

        This method is called recursively for each module in the base model during
        the PEFT application process (`apply`). Subclasses must implement this
        to define how individual modules (especially target modules identified using
        the subclass's configuration) are modified (e.g., wrapped with adapter layers).

        Args:
            module (nn.Module): The individual module to be potentially transformed.
            name (Optional[str]): The name of the module within the model structure.
            prefix (Optional[str]): A prefix indicating the nesting hierarchy.

        Returns:
            nn.Module: The transformed module (potentially the original module if no
                       transformation is applied).
        """
        raise NotImplementedError("Subclasses must implement the 'transform' method.")


    def __call__(self, model: list[MegatronModule]) -> list[MegatronModule]:
        """Applies the PEFT transformation to the entire model.

        This method freezes the model parameters and then walks through the
        model structure, applying the `transform` method (implemented by the subclass)
        to each submodule within each model chunk.

        Args:
            model (List[MegatronModule]): The base Megatron model
                (or list of model chunks for pipeline parallelism) to be fine-tuned.
            **kwargs: Additional keyword arguments for the transform method.

        Returns:
            List[MegatronModule]: The transformed model with PEFT applied.
        """
        self.freeze_model(model)

        for model_chunk in model:
            walk(model_chunk, self.transform)

        if not self.training:
            self.freeze(model)
        
        self.set_params_to_save(model)

        return model


    def freeze_model(self, model: list[MegatronModule]) -> None:
        """Freezes all parameters of the base model.

        Sets `requires_grad=False` for all parameters. Subclasses can override
        this method to implement custom freezing strategies (e.g., keeping
        embeddings or layer norms trainable).

        Args:
            model (List[MegatronModule]): The list of base model chunks.
        """
        for model_chunk in model:
            model_chunk.freeze()

        if self.training:
            for model_chunk in model:
                model_chunk.train(mode=True)


    def set_params_to_save(self, model: list[MegatronModule]) -> None:
        """Identifies and sets the names of the trainable parameters (adapter parameters).

        This method iterates through all parameters in the model (across all chunks)
        and returns the names of those where `requires_grad` is True. It caches the
        result after the first call.

        Args:
            model (List[MegatronModule]): The PEFT-transformed model
                (or list of model chunks).
        """
        names = set()
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if param.requires_grad:
                    names.add(name)
            for module_name, module in model_chunk.named_modules():
                if hasattr(module, "track_running_stats"):
                    for buffer_name, buffer in module.named_buffers():
                        if buffer is not None:
                            names.add(module_name + "." + buffer_name)
        self.params_to_save = names


    def adapter_key_filter(self, key: str | tuple) -> bool:
        """
        Given a key in the state dict, return whether the key is an adapter (or base model).
        This function can be subclassed in each PEFT method class.
        """
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters")
