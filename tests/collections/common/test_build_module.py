# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""build_module: instantiating a configurable sub-module from a class, a partial or a Hydra section."""

import functools

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.common.parts.utils import SquaredReLU, activation_registry, block_binds, build_module


class TestBuildModule:
    @pytest.mark.unit
    def test_class_gets_the_parent_defaults(self):
        norm = build_module(torch.nn.LayerNorm, normalized_shape=8)
        assert isinstance(norm, torch.nn.LayerNorm) and norm.normalized_shape == (8,) and norm.eps == 1e-5

    @pytest.mark.unit
    def test_partial_keywords_override_the_parent_defaults(self):
        block = functools.partial(torch.nn.LayerNorm, eps=1e-6, elementwise_affine=False)
        norm = build_module(block, normalized_shape=8, eps=1e-5)
        assert norm.eps == 1e-6 and norm.weight is None and norm.normalized_shape == (8,)

    @pytest.mark.unit
    def test_hydra_section_is_instantiated_lazily(self):
        section = DictConfig({"_target_": "torch.nn.LayerNorm", "eps": 1e-6})
        norm = build_module(section, normalized_shape=8)
        assert isinstance(norm, torch.nn.LayerNorm) and norm.eps == 1e-6
        explicit = {"_target_": "torch.nn.LayerNorm", "_partial_": True, "eps": 1e-3}
        assert build_module(explicit, normalized_shape=4).eps == 1e-3

    @pytest.mark.unit
    def test_section_targets_are_validated(self):
        with pytest.raises(Exception, match="unsafe target"):
            build_module({"_target_": "os.system"}, command="true")

    @pytest.mark.unit
    def test_block_binds(self):
        assert block_binds(functools.partial(torch.nn.LayerNorm, eps=1e-6), "eps")
        assert not block_binds(functools.partial(torch.nn.LayerNorm), "eps")
        assert block_binds({"_target_": "torch.nn.LayerNorm", "eps": 1e-6}, "eps")
        assert not block_binds(torch.nn.LayerNorm, "eps") and not block_binds(None, "eps")

    @pytest.mark.unit
    def test_squared_relu(self):
        x = torch.tensor([-2.0, 0.0, 3.0])
        torch.testing.assert_close(SquaredReLU()(x), torch.tensor([0.0, 0.0, 9.0]))
        assert activation_registry["squared_relu"] is SquaredReLU
