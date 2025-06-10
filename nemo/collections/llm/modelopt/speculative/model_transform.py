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

from typing import Any, Optional

import torch.nn as nn

from nemo.utils import logging
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")
mtsp, _ = safe_import("modelopt.torch.speculative")


class SpeculativeTransform:
    """
    A callable class that applies speculative decoding transformation to a model.

    This transform applies modelopt's speculative decoding techniques to a model
    when called. It can be used directly as a model_transform parameter in NeMo models.

    Args:
        algorithm: The algorithm to use for speculative decoding.
        config: The configuration for the algorithm.
            (See https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/speculative/config.py)

    Example:
        >>> from nemo.collections.llm.modelopt import SpeculativeTransform
        >>> model = GPTModel(config, model_transform=SpeculativeTransform())
        >>> # The model will be transformed when trainer.fit() or trainer.validate() is called
    """

    ALGORITHMS = ["eagle"]  # more TBD

    def __init__(self, algorithm: str = "eagle", config: Optional[dict[str, Any]] = None):
        if not HAVE_MODELOPT:
            raise ImportError("nvidia-modelopt is required to use SpeculativeTransform")

        assert algorithm in self.ALGORITHMS, f"Invalid algorithm: {algorithm}. Choices: {self.ALGORITHMS}"
        if config is None:
            config = mtsp.EAGLE3_DEFAULT_CFG["config"]

        self.algorithm = algorithm
        self.config = config

    def __call__(self, model: nn.Module) -> nn.Module:
        """
        Apply speculative decoding transformation to the model.

        Args:
            model: The model to transform.

        Returns:
            The transformed model.
        """
        unwrapped_model = unwrap_model(model)

        # Verify model is compatible with speculative decoding
        assert hasattr(unwrapped_model, "config"), "Model must have a config attached."
        if unwrapped_model.config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("SpeculativeTransform is incompatible with virtual pipeline parallelism.")
        if unwrapped_model.config.gradient_accumulation_fusion is True:
            raise ValueError("SpeculativeTransform is incompatible with gradient accumulation fusion.")

        # Check if the model has already been transformed with speculative decoding
        if mto.ModeloptStateManager.has_state_for_mode_type("speculative", model=unwrapped_model):
            logging.info("Model has already been transformed with speculative decoding. Skipping transformation.")
            return model

        logging.info(
            f"Converting to Speculative Decoding model with algorithm: {self.algorithm} and config:\n{self.config}"
        )
        mtsp.convert(  # assumes in-place
            unwrapped_model,
            [(self.algorithm, self.config)],
        )

        return model
