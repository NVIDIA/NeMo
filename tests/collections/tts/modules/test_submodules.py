# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.modules import submodules 

@pytest.mark.unit
def test_conditional_layer_norm():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    lm = torch.nn.LayerNorm(embedding_dim)
    clm = submodules.ConditionalLayerNorm(embedding_dim)
    assert torch.all(lm(embedding) == clm(embedding))
    
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    image = torch.randn(N, C, H, W)
    lm = torch.nn.LayerNorm([C, H, W])
    clm = submodules.ConditionalLayerNorm([C, H, W])
    assert torch.all(lm(image) == clm(image))