# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,


# limitations under the License.
# =============================================================================
from abc import ABC, abstractmethod

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Dict, Optional

from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult
from nemo.utils import instantiate_class_from_config


def assign_neural_type(t, neural_type: NeuralType):
    t.neural_type = neural_type
    return t


def check_neural_type(t, neural_type: NeuralType):
    if hasattr(t, 'neural_type'):
        return neural_type.compare(t.neural_type)
    else:
        return NeuralTypeComparisonResult.UNCHECKED


def compare_input_types(input_types, in_dict):
    return None


def attach_output_types(input_types, out_dict):
    return None


class NeuralModuleAPI(ABC):
    """
        Abstract class offering interface shared between all Neural Modules.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def typed_forward(self, **kwargs):
        compare_input_types(self.input_types, **kwargs)
        result = self.forward(**kwargs)
        attach_output_types(self.output_types, **kwargs)
        return result

    @classmethod
    def from_config(cls, configuration: Dict[str, Any], name: str = None, overwrite_params: Dict[str, Any] = {}):
        return instantiate_class_from_config(configuration, name, overwrite_params)


class NeuralModelAPI(NeuralModuleAPI):
    @classmethod
    @abstractmethod
    def from_cloud(cls, name: str):
        """
        Instantiates an instance of Neural Module from NVIDIA NGC cloud
        Args:
            name: string key which will be used to find the module

        Returns:
            and instance of a class derived from NeuralModuleAPI
        """
        pass
