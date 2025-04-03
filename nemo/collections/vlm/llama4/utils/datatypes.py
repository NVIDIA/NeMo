# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Follwoing are temporarily adapted and modified from:
# https://github.com/meta-llama/llama-stack/tree/main/llama_stack/models/llama

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

DUMMY_CHECKPOINT = False


@dataclass
class MaskedEmbedding:
    embedding: torch.Tensor
    mask: torch.Tensor


@dataclass
class LLMInput:
    """
    This is the input to the LLM from the "user" -- the user in this case views the
    Llama4 model holistically and does not care or know about its inner workings (e.g.,
    whether it has an encoder or if it is early fusion or not.)

    This is distinct from the "CoreTransformerInput" class which is really the Llama4
    backbone operating on early fused modalities and producing text output
    """

    tokens: torch.Tensor

    # images are already pre-processed (resized, tiled, etc.)
    images: Optional[List[torch.Tensor]] = None


@dataclass
class CoreTransformerInput:
    """
    This is the "core" backbone transformer of the Llama4 model. Inputs for other modalities
    are expected to be "embedded" via encoders sitting before this layer in the model.
    """

    tokens: torch.Tensor

    # tokens_position defines the position of the tokens in each batch,
    # - when it is a tensor ([batch_size,]), it is the start position of the tokens in each batch
    # - when it is an int, the start position are the same for all batches
    tokens_position: Union[torch.Tensor, int]
    image_embedding: Optional[MaskedEmbedding] = None


@dataclass
class LLMOutput:
    logits: torch.Tensor


CoreTransformerOutput = LLMOutput
