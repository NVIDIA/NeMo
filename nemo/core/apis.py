# Copyright (c) 2019-, NVIDIA CORPORATION. All rights reserved.
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

from abc import ABC, abstractmethod
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
    if input_types is not None:
        for key, value in in_dict.items():
            if (
                hasattr(value, 'neural_type')
                and input_types[key].compare(value.neural_type) != NeuralTypeComparisonResult.SAME
            ):
                raise TypeError(f"{input_types[key].compare(value.neural_type)}")
    # else:
    # logging.warning(f"typed_forward was used but input types were not specified")


def attach_output_types(output_types, out_dict):
    if output_types is not None:
        out_types_list = list(output_types.items())
        if not isinstance(out_dict, tuple) and not isinstance(out_dict, list):
            out_dict.neural_type = out_types_list[0][1]
        else:
            for ind, res in enumerate(out_dict):
                res.neural_type = out_types_list[ind][1]


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
        compare_input_types(input_types=self.input_types, in_dict=kwargs)
        result = self.forward(**kwargs)
        attach_output_types(output_types=self.output_types, out_dict=result)
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

    @abstractmethod
    def setup_training_data(self, train_data_layer_params):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_params: training data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def save_to(self, save_path: str, optimize_for_deployment=True):
        pass

    @classmethod
    @abstractmethod
    def restore_from(cls, restore_path: str):
        pass
