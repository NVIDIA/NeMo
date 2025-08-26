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

import abc
from typing import Set, Optional, List, Dict
import torch
import torch.nn as nn
# Conditional import for MegatronModule type hinting
try:
    from megatron.core.transformer import MegatronModule
except ImportError:
    # Define a fallback or handle the absence of MegatronModule if necessary
    MegatronModule = nn.Module # Fallback type

from nemo.tron.utils.common_utils import print_rank_0


class TronPeftTransform(abc.ABC):
    """
    Abstract base class for applying Parameter-Efficient Fine-Tuning (PEFT)
    transformations within the NeMo TRON framework.

    Operates on the model list returned by `get_model_from_config` *before*
    DDP wrapping.

    Workflow:
    1. Instantiate a subclass (e.g., `LoraTronPeftTransform`).
    2. Call the instance with the model list: `transformed_model_list = peft_instance(model_list)`.
       This applies the `transform` to each module and freezes base weights.
    3. Retrieve the trainable parameters: `trainable_params = peft_instance.get_trainable_params(transformed_model_list)`.
    """
    def __init__(self):
        self.trainable_params_cache: Optional[Set[nn.Parameter]] = None
        self._already_applied = False

    @abc.abstractmethod
    def transform(self, module: nn.Module, name: str, prefix: str) -> Optional[nn.Module]:
        """
        Core PEFT logic for transforming a single module.

        If the given module (identified by name and prefix) is a target for this
        PEFT method, this method should return the transformed module (e.g.,
        the original module wrapped with adapters, or a completely new module).
        The parameters added by the PEFT method (e.g., adapter weights) should
        be marked, for example, by setting `param.is_adapter = True`.

        If the module is not a target, this method should return `None`.

        Args:
            module: The module instance to potentially transform.
            name: The name of the module relative to its parent.
            prefix: The fully qualified prefix leading to this module.

        Returns:
            The transformed nn.Module if applicable, otherwise None.
        """
        pass

    def __call__(self, model_list: List[MegatronModule]) -> List[MegatronModule]:
        """
        Applies the PEFT transformation to the entire model (list of modules)
        in-place by walking through its modules and applying the `transform` method.
        It also freezes the original base model parameters afterwards.

        Prevents re-application if called multiple times.

        Args:
            model_list: The list of MegatronModules returned by `get_model_from_config`.

        Returns:
            The transformed list of MegatronModules (modified in-place).
        """
        if self._already_applied:
            print_rank_0(f"Warning: {self.__class__.__name__} PEFT transform already applied. Skipping.")
            return model_list

        print_rank_0(f"Applying {self.__class__.__name__} transformation...")
        trainable_params = set()
        all_modified_modules: Dict[str, nn.Module] = {} # Track modified modules globally

        for i, model_chunk in enumerate(model_list):
            print_rank_0(f" Processing model chunk {i}...")
            modified_parents_in_chunk = {} # Track modifications within the chunk

            # Use model_chunk.named_modules() which should handle MCore structure
            # Use list() to prevent issues if the structure is modified during iteration
            modules_to_process = list(model_chunk.named_modules())

            for name, module in modules_to_process:
                 # Skip root module (prefix='') and already processed subtrees
                if not name or any(name.startswith(parent_name + '.') for parent_name in modified_parents_in_chunk):
                    continue

                # Determine prefix and local name
                if '.' in name:
                    prefix, local_name = name.rsplit('.', 1)
                else:
                    prefix = ''
                    local_name = name

                transformed_module = self.transform(module, local_name, prefix)

                if transformed_module is not None:
                    try:
                        parent_module = model_chunk.get_submodule(prefix) if prefix else model_chunk
                        setattr(parent_module, local_name, transformed_module)
                        # Use the *global* name for tracking overall modifications
                        global_module_name = f"chunk_{i}.{name}" if name else f"chunk_{i}"
                        print_rank_0(f"  Replaced {global_module_name} with {transformed_module.__class__.__name__}")
                        modified_parents_in_chunk[name] = True
                        all_modified_modules[global_module_name] = transformed_module # Store globally

                        # Collect trainable parameters
                        for param in transformed_module.parameters():
                            if getattr(param, 'is_adapter', False):
                                trainable_params.add(param)
                    except Exception as e: # Catch broader exceptions during setattr/get_submodule
                         print_rank_0(f"  ERROR applying transform to {name} in chunk {i}: {e}")
                         # Optionally raise e

        # Freeze parameters *after* all transformations are done
        self.freeze_model(model_list, trainable_params)
        self.trainable_params_cache = trainable_params
        self._already_applied = True
        print_rank_0(f"{self.__class__.__name__} applied. Found {len(trainable_params)} trainable parameters.")

        return model_list

    def freeze_model(self, model_list: List[MegatronModule], trainable_params: Set[nn.Parameter]):
        """
        Freezes non-trainable parameters and restores original training mode.

        Args:
            model_list: The list of models (potentially transformed).
            trainable_params: The set of parameters that should remain trainable.
        """
        print_rank_0("Freezing base model parameters...")
        frozen_count = 0
        trainable_count = 0
        total_param_count = 0

        # Store original training modes
        original_modes = [chunk.training for chunk in model_list]

        # Set requires_grad flags
        all_params = [p for chunk in model_list for p in chunk.parameters()]
        for param in all_params:
            total_param_count += 1
            if param in trainable_params:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        # Restore original training modes
        for i, chunk in enumerate(model_list):
            chunk.train(mode=original_modes[i])

        print_rank_0(f"Froze {frozen_count}/{total_param_count} parameters. Kept {trainable_count} parameters trainable. Restored training modes.")
        # Note: Buffer freezing is not handled here as requires_grad doesn't apply.
        # Saving adapter-specific buffers relies on saving the state_dict of adapter modules.

    def get_trainable_params(self, model_list: List[MegatronModule]) -> Set[nn.Parameter]:
        """
        Returns the set of trainable parameters identified during the apply step.
        (Implementation remains the same as previous version - relies on cache)
        """
        if not self._already_applied or self.trainable_params_cache is None:
             print_rank_0("Warning: get_trainable_params called before __call__ completed or cache is missing. Recalculating...")
             self.trainable_params_cache = set()
             all_params = [p for chunk in model_list for p in chunk.parameters()]
             for p in all_params:
                  if p.requires_grad: # Check current state
                       self.trainable_params_cache.add(p)
             if not self.trainable_params_cache:
                  print_rank_0("Warning: Recalculation found no trainable parameters.")
        return self.trainable_params_cache

    # Note on Buffers:
    # Saving buffers associated with PEFT adapters (if any exist beyond standard layers)
    # needs to be handled during checkpointing. The recommended approach is to save
    # the state_dict of the adapter modules themselves, which will include their buffers.
    # The `_generate_optimizer_sd` helper or the `PeftSaveStrategyWrapper` (depending
    # on the chosen checkpointing option) must correctly identify and include the
    # state corresponding to these adapter modules/parameters. 