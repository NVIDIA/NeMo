# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

__all__ = [
    'ElementType',
    'VoidType',
    'ChannelType',
    'AcousticEncodedRepresentation',
    'AudioSignal',
    'SpectrogramType',
    'MelSpectrogramType',
    'MFCCSpectrogramType',
    'LogitsType',
    'LabelsType',
    'LossType',
    'RegressionValuesType',
    'CategoricalValuesType',
    'PredictionsType',
    'LogprobsType',
    'LengthsType',
    'EmbeddedTextType',
    'EncodedRepresentation'
]
import abc
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from .comparison import NeuralTypeComparisonResult


class ElementType(ABC):
    """Abstract class defining semantics of the tensor elements.
    We are replying on Python for inheritance checking"""

    @abstractmethod
    def __str__(cls):
        pass

    @property
    def type_parameters(self) -> Dict:
        """Override this property to parametrize your type"""
        return {}

    @property
    def fields(self) -> Optional[Tuple]:
        return None

    def compare(self, second) -> NeuralTypeComparisonResult:
        # First, check general compatibility
        first_t = type(self)
        second_t = type(second)

        if first_t == second_t:
            result = NeuralTypeComparisonResult.SAME
        elif issubclass(first_t, second_t):
            result = NeuralTypeComparisonResult.LESS
        elif issubclass(second_t, first_t):
            result = NeuralTypeComparisonResult.GREATER
        else:
            result = NeuralTypeComparisonResult.INCOMPATIBLE

        if result != NeuralTypeComparisonResult.SAME:
            return result
        else:
            # now check that all parameters match
            check_params = set(self.type_parameters.keys()) == set(second.type_parameters.keys())
            if check_params is False:
                return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            else:
                for k1, v1 in self.type_parameters.items():
                    if v1 != second.type_parameters[k1]:
                        return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            # check that all fields match
            if self.fields == second.fields:
                return NeuralTypeComparisonResult.SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE


class VoidType(ElementType):
    """Void-like type which is compatible with everything
    """

    def __str__(self):
        return str("void type. compatible with everything")

    def compare(cls, second: abc.ABCMeta) -> NeuralTypeComparisonResult:
        return NeuralTypeComparisonResult.SAME


# TODO: Consider moving these files elsewhere
class ChannelType(ElementType):
    def __str__(self):
        return "convolutional channel value"


class EmbeddedTextType(ChannelType):
    def __str__(self):
        return "text embedding"


class LogitsType(ElementType):
    def __str__(self):
        return "neural type representing logits"


class LogprobsType(ElementType):
    def __str__(self):
        return "neural type representing log probabilities"


class LabelsType(ElementType):
    def __str__(self):
        return "neural type representing labels"


class LengthsType(ElementType):
    def __str__(self):
        return "neural type representing lengths of something"


class LossType(ElementType):
    def __str__(self):
        return "neural type representing loss value"


class EncodedRepresentation(ChannelType):
    def __str__(self):
        return "encoded representation, for example, encoder's output"


class AcousticEncodedRepresentation(EncodedRepresentation):
    def __str__(self):
        return "encoded representation returned by the acoustic encoder model"


class AudioSignal(ElementType):
    def __str__(self):
        return "encoded representation returned by the acoustic encoder model"

    def __init__(self, freq=16000):
        self._params = {}
        self._params['freq'] = freq

    @property
    def type_parameters(self):
        return self._params


class SpectrogramType(ChannelType):
    def __str__(self):
        return "generic spectorgram type"


class MelSpectrogramType(SpectrogramType):
    def __str__(self):
        return "mel spectorgram type"


class MFCCSpectrogramType(SpectrogramType):
    def __str__(self):
        return "mfcc spectorgram type"


class PredictionsType(ElementType):
    def __str__(self):
        return "predictions values type"


class RegressionValuesType(PredictionsType):
    def __str__(self):
        return "regression values type"


class CategoricalValuesType(PredictionsType):
    def __str__(self):
        return "regression values type"
