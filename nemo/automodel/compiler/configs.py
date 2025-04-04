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

from dataclasses import dataclass, field


@dataclass
class TorchCompileConfig:
    """Config for torch.compile
    Options:
    - module_selector (str): reg-exp to match modules to compile, useful for multi-trunk
      models where you want to apply it on one of them only. If empty will apply transform to root
      module.
    - apply_pre_wrap: if True will compile before wrapping with DDP/FSDP2 & vice-versa.
    - kwargs (dict): kwargs to pass to torch.compile.
    """

    module_selector: str = ''
    apply_pre_wrap: bool = True
    kwargs: dict = field(default_factory=dict)


@dataclass
class ThunderConfig:
    """Config for Thunder
    Options:
    - module_selector (str): reg-exp to match modules to compile, useful for multi-trunk
      models where you want to apply it on one of them only. If empty will apply transform to root
      module.
    - apply_pre_wrap: if True will compile before wrapping with DDP/FSDP2 & vice-versa.
    - kwargs (dict): kwargs to pass to thunder, currently unused.
    - profile (bool): toggle for thunder's profiler.
    """

    module_selector: str = ''
    apply_pre_wrap: bool = True
    kwargs: dict = field(default_factory=dict)
    profile: bool = False
