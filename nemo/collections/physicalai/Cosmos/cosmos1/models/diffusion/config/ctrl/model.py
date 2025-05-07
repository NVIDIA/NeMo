# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import attrs

from cosmos1.models.diffusion.config.base.model import DefaultModelConfig
from cosmos1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class CtrlModelConfig(DefaultModelConfig):
    net_ctrl: LazyDict = None
    hint_key: str = None
    base_load_from: LazyDict = None
    finetune_base_model: bool = False
    hint_mask: list = [True]
    hint_dropout_rate: float = 0.0
    num_control_blocks: int = 5
    random_drop_control_blocks: bool = False
    pixel_corruptor: LazyDict = None
