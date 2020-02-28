# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

from typing import Dict, List, Optional, Set, Tuple

from nemo.core import NeuralModule, NeuralType, WeightShareTransform


class CompositeNeuralModule(NeuralModule):
    def __init__(
        self,
        modules: List[NeuralModule],
        pipeline,
        input_ports: Optional[Dict[str, NeuralType]] = None,
        output_ports: Optional[Dict[str, NeuralType]] = None,
    ):
        self.modules = modules
        self.pipeline = pipeline
        if input_ports is None:
            self.__input_ports = modules[0].input_ports
        else:
            self.__input_ports = input_ports

        if output_ports is None:
            self.__output_ports = modules[-1].output_ports
        else:
            self.__output_ports = output_ports

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self.__input_ports

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self.__output_ports

    def get_weights(self) -> Optional[Dict[(str, bool)]]:
        raise NotImplemented("Not implemented for composite modules")

    def set_weights(
        self,
        name2weight: Dict[(str, Tuple[str, bool])],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        raise NotImplemented("Not implemented for composite modules")

    def tie_weights_with(
        self,
        module,
        weight_names=List[str],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        raise NotImplemented("Not implemented for composite modules")

    def save_to(self, path: str):
        # TODO
        pass

    def restore_from(self, path: str):
        # TODO
        pass

    def freeze(self, weights: Set[str] = None):
        if weights is None:
            for module in self.modules:
                module.freeze()
        else:
            raise NotImplemented("For composite modules, freeze only works for all weights (e.g. weights=None)")

    def unfreeze(self, weights: Set[str] = None):
        if weights is None:
            for module in self.modules:
                module.unfreeze()
        else:
            raise NotImplemented("For composite modules, unfreeze only works for all weights (e.g. weights=None)")

    @property
    def num_weights(self):
        num_weights = 0
        for module in self.modules:
            num_weights += module.num_weights
