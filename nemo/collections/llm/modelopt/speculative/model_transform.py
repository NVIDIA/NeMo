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

from nemo.collections.llm import GPTModel
from nemo.utils import logging
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")
mtsp, _ = safe_import("modelopt.torch.speculative")


ALGORITHMS = {
    "eagle3": mtsp.EAGLE3_DEFAULT_CFG,
    # more TBD
}


def apply_speculative_decoding(model: nn.Module, algorithm: str = "eagle3") -> nn.Module:
    """
    Transform a model to enable speculative decoding using Model Optimizer.

    Args:
        model: The model to transform.
        algorithm: The algorithm to use for speculative decoding.
            (See https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/speculative/config.py)

    Returns:
        The transformed model.
    """
    if not HAVE_MODELOPT:
        raise ImportError("nvidia-modelopt is required to use speculative decoding")

    assert algorithm in ALGORITHMS, f"Invalid algorithm: {algorithm}. Choices: {ALGORITHMS.keys()}"
    algo_cfg = ALGORITHMS[algorithm]

    assert isinstance(model, GPTModel), "Speculative Decoding currently only supported for GPT models."
    unwrapped_model = unwrap_model(model)

    # Check if the model has already been transformed with speculative decoding
    if mto.ModeloptStateManager.has_state_for_mode_type("speculative", model=unwrapped_model):
        logging.info("Model has already been transformed with speculative decoding. Skipping transformation.")
        return model

    # Verify model is compatible with speculative decoding
    assert hasattr(unwrapped_model, "config"), "Model must have a config attached."
    if unwrapped_model.config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Speculative decoding is incompatible with virtual pipeline parallelism.")

    algo, cfg = algo_cfg["algorithm"], algo_cfg["config"]
    logging.info(f"Converting to Speculative Decoding model with algorithm: {algo} and config:\n{cfg}")
    mtsp.convert(unwrapped_model, [(algo, cfg)])  # assumes in-place

    return model
