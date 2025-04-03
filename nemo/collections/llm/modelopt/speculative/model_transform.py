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

import torch.nn as nn

from nemo.utils import logging
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")
mtsp, _ = safe_import("modelopt.torch.speculative")


class SpeculativeTransform:
    """
    A callable class that applies speculative decoding transformation to a model.

    This transform applies modelopt's speculative decoding techniques (Medusa or Eagle) to a model
    when called. It can be used directly as a model_transform parameter in NeMo models.

    Args:
        medusa_heads: Number of Medusa heads to use for speculative decoding.
        eagle_layers: Number of Eagle layers to use for speculative decoding.

    Example:
        >>> from nemo.collections.llm.modelopt.speculative.model_transform import SpeculativeTransform
        >>> model = GPTModel(config, model_transform=SpeculativeTransform(num_medusa_heads=4))
        >>> # The model will be transformed when trainer.fit() or trainer.validate() is called
    """

    def __init__(self, num_medusa_heads: int = 0, num_eagle_layers: int = 0):
        if not HAVE_MODELOPT:
            raise ImportError("nvidia-modelopt is required to use SpeculativeTransform")
        if num_medusa_heads > 0 and num_eagle_layers > 0:
            raise ValueError("Only one of num_medusa_heads or num_eagle_layers should be specified, not both.")
        if num_medusa_heads == 0 and num_eagle_layers == 0:
            raise ValueError("At least one of num_medusa_heads or num_eagle_layers must be greater than 0.")

        self.num_medusa_heads = num_medusa_heads
        self.num_eagle_layers = num_eagle_layers
        # Default value for Medusa layers if using Medusa heads
        self.num_medusa_layers = 1

    def __call__(self, model: nn.Module) -> nn.Module:
        """
        Apply speculative decoding transformation to the model.

        Args:
            model: The model to transform.

        Returns:
            The transformed model.
        """
        unwrapped_model = unwrap_model(model)
        # Check if the model has already been transformed with speculative decoding
        if mto.ModeloptStateManager.has_state_for_mode_type("speculative", model=unwrapped_model):
            logging.info("Model has already been transformed with speculative decoding. Skipping transformation.")
            return model

        # Check for incompatible model configurations
        if hasattr(unwrapped_model, "config"):
            if unwrapped_model.config.virtual_pipeline_model_parallel_size is not None:
                raise ValueError("SpeculativeTransform is incompatible with virtual pipeline parallelism.")
            if unwrapped_model.config.moe_grouped_gemm is True:
                logging.warning("Disabling MOE grouped GEMM for SpeculativeTransform.")
                unwrapped_model.config.moe_grouped_gemm = False

        if self.num_medusa_heads > 0:
            logging.info(f"Converting to Speculative Decoding model with num_medusa_heads={self.num_medusa_heads}")
            sp_config = {"medusa_num_heads": self.num_medusa_heads, "medusa_num_layers": self.num_medusa_layers}
            mtsp.convert(  # assumes in-place
                unwrapped_model,
                [("medusa", sp_config)],
            )
        elif self.num_eagle_layers > 0:
            logging.info(f"Converting to Speculative Decoding model with num_eagle_layers={self.num_eagle_layers}")
            sp_config = {"eagle_num_layers": self.num_eagle_layers}
            mtsp.convert(  # assumes in-place
                unwrapped_model,
                [("eagle", sp_config)],
            )

        return model
