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

from abc import ABC

import torch

from nemo.core import NeuralTypeComparisonResult
from nemo.core.apis.common import NeuralModuleAPI

__all__ = ['NeuralModulePT']


class NeuralModulePT(torch.nn.Module, NeuralModuleAPI, ABC):
    """
    Abstract class offering interface shared between all PyTorch Neural Modules.
    """

    def typed_forward(self, **kwargs):
        # TODO: Consider if this can be turned into decorator for __call__ or forward
        self.__validate_input_types(in_objects=kwargs)
        result = self.forward(**kwargs)
        self.__attach_and_validate_output_types(out_objects=result)
        return result

    def __validate_input_types(self, in_objects):
        # TODO: Properly implement this
        if self.input_types is not None:
            for key, value in in_objects.items():
                if (
                    hasattr(value, 'neural_type')
                    and self.input_types[key].compare(value.neural_type) != NeuralTypeComparisonResult.SAME
                ):
                    raise TypeError(f"{self.input_types[key].compare(value.neural_type)}")

    def __attach_and_validate_output_types(self, out_objects):
        # TODO: Properly implement this
        if self.output_types is not None:
            out_types_list = list(self.output_types.items())
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                out_objects.neural_type = out_types_list[0][1]
            else:
                for ind, res in enumerate(out_objects):
                    res.neural_type = out_types_list[ind][1]
