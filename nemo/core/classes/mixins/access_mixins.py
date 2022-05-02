# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf import DictConfig
from abc import ABC
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Optional, Union

import torch

_ACCESS_CFG = DictConfig({"access_all_intermediate": False,
                          "detach": False,
                          "convert_to_cpu": False
                          })

def set_access_cfg(cfg: 'DictConfig'):
    global _ACCESS_CFG
    _ACCESS_CFG = cfg

class AccessMixin(ABC):
    """
    Allows access to output of intermediate layers of a model
    """

    def __init__(self):
        super().__init__()
        self._registry = []

    def register_accessible_tensor(
            self, tensor
    ):
        if self.access_cfg.get('convert_to_cpu', False):
            tensor = tensor.cpu()

        if self.access_cfg.get('detach', False):
            tensor = tensor.detach()

        if not hasattr(self, '_registry'):
            self._registry = []

        self._registry.append(tensor)

    @classmethod
    def get_module_registry(
            cls, module: torch.nn.Module
    ):
        """
        Extract all registries from named submodules, return dictionary where
        the keys are the flattened module names, the values are the internal registry
        of each such module.
        """
        module_registry = {}
        for name, m in module.named_modules():
            if hasattr(m, '_registry') and len(m._registry) > 0:
                module_registry[name] = m._registry
        return module_registry

    def reset_registry(self):
        """
        Reset the registries of all named sub-modules
        """
        if hasattr(self, "_registry"):
            self._registry.clear()
        for _, m in self.named_modules():
            if hasattr(m, "_registry"):
                m._registry.clear()

    @property
    def access_cfg(self):
        """
        Returns:
            The global access config shared across all access mixin modules.
        """
        global _ACCESS_CFG
        return _ACCESS_CFG