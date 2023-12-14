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

from abc import ABC
from typing import Any, Dict, Optional

import torch

from nemo.core.neural_types.comparison import NeuralTypeComparisonResult

__all__ = [
    'ElementType',
    'VoidType',
    'BoolType',
    'ChannelType',
    'AcousticEncodedRepresentation',
    'AudioSignal',
    'VideoSignal',
    'SpectrogramType',
    'MelSpectrogramType',
    'MFCCSpectrogramType',
    'LogitsType',
    'LabelsType',
    'HypothesisType',
    'LossType',
    'RegressionValuesType',
    'CategoricalValuesType',
    'PredictionsType',
    'LogprobsType',
    'ProbsType',
    'LengthsType',
    'EmbeddedTextType',
    'EncodedRepresentation',
    'MaskType',
    'Target',
    'ClassificationTarget',
    'ImageFeatureValue',
    'Index',
    'ImageValue',
    'NormalizedImageValue',
    'StringLabel',
    'StringType',
    'TokenIndex',
    'Length',
    'IntType',
    'FloatType',
    'NormalDistributionSamplesType',
    'NormalDistributionMeanType',
    'NormalDistributionLogVarianceType',
    'TokenDurationType',
    'TokenLogDurationType',
    'LogDeterminantType',
    'SequenceToSequenceAlignmentType',
]


class ElementType(ABC):
    """Abstract class defining semantics of the tensor elements.
    We are relying on Python for inheritance checking"""

    def __str__(self):
        if torch.jit.is_scripting():
            return "SuppressedForTorchScript"
        return self.__doc__

    def __repr__(self):
        if torch.jit.is_scripting():
            return "SuppressedForTorchScript"
        return self.__class__.__name__

    @property
    def type_parameters(self) -> Dict[str, Any]:
        """Override this property to parametrize your type. For example, you can specify 'storage' type such as
        float, int, bool with 'dtype' keyword. Another example, is if you want to represent a signal with a
        particular property (say, sample frequency), then you can put sample_freq->value in there.
        When two types are compared their type_parameters must match."""
        return {}

    @property
    def fields(self):
        """This should be used to logically represent tuples/structures. For example, if you want to represent a
        bounding box (x, y, width, height) you can put a tuple with names ('x', y', 'w', 'h') in here.
        Under the hood this should be converted to the last tesnor dimension of fixed size = len(fields).
        When two types are compared their fields must match."""
        return None

    def compare(self, second) -> NeuralTypeComparisonResult:
        if torch.jit.is_scripting():
            # Suppress in torch.jit.script
            return NeuralTypeComparisonResult.SAME
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
                    if v1 is None or second.type_parameters[k1] is None:
                        # Treat None as Void
                        continue
                    if v1 != second.type_parameters[k1]:
                        return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            # check that all fields match
            if self.fields == second.fields:
                return NeuralTypeComparisonResult.SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE


class VoidType(ElementType):
    """Void-like type which is compatible with everything.
    It is a good practice to use this type only as necessary.
    For example, when you need template-like functionality.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass

    def compare(cls, second) -> NeuralTypeComparisonResult:
        return NeuralTypeComparisonResult.SAME


# TODO: Consider moving these files elsewhere
class ChannelType(ElementType):
    """Element to represent convolutional input/output channel.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class EmbeddedTextType(ChannelType):
    """Element to represent output on word/text embedding layers
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class LogitsType(ElementType):
    """Element type to represent logits"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class ProbsType(ElementType):
    """Element type to represent probabilities. For example, outputs of softmax layers."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class LogprobsType(ElementType):
    """Element type to represent log-probabilities. For example, outputs of log softmax layers."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class LabelsType(ElementType):
    """Element type to represent some sort of labels. This is often used as a base class to create
    a more concrete types such as RegressionValuesType, etc."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class HypothesisType(LabelsType):
    """Element type to represent some decoded hypothesis, which may further be processed to obtain
    a concrete label."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class LengthsType(ElementType):
    """Element type representing lengths of something"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class LossType(ElementType):
    """Element type to represent outputs of Loss modules"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class EncodedRepresentation(ChannelType):
    """Element type to represent encoded representation, for example, encoder's output"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class AcousticEncodedRepresentation(EncodedRepresentation):
    """Element type to represent encoded representation returned by the acoustic encoder model"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class AudioSignal(ElementType):
    """Element type to represent encoded representation returned by the acoustic encoder model
    Args:
        freq (int): sampling frequency of a signal. Note that two signals will only be the same if their
        freq is the same.
    """

    def __init__(self, freq: Optional[int] = None):
        self._params: Dict[str, Any] = {}
        self._params['freq'] = freq

    @property
    def type_parameters(self):
        return self._params


class VideoSignal(ElementType):
    """Element type to represent encoded representation returned by the visual encoder model
    Args:
        fps (int): frames per second.
    """

    def __init__(self, fps: Optional[int] = None):
        self._params: dict[str, Any] = {}
        self._params['fps'] = fps

    @property
    def type_parameters(self):
        return self._params


class SpectrogramType(ChannelType):
    """Element type to represent generic spectrogram signal"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class MelSpectrogramType(SpectrogramType):
    """Element type to represent mel spectrogram signal"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class MFCCSpectrogramType(SpectrogramType):
    """Element type to represent MFCC spectrogram signal"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class PredictionsType(LabelsType):
    """Element type to represent some sort of predictions returned by model"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class RegressionValuesType(PredictionsType):
    """Element type to represent labels for regression task"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class CategoricalValuesType(PredictionsType):
    """Element type to represent labels for categorical classification task"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class MaskType(PredictionsType):
    """Element type to represent a boolean mask"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class Index(ElementType):
    """Type representing an element being an index of the sample."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class Target(ElementType):
    """
        Type representing an element being a target value.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class ClassificationTarget(Target):
    """
        Type representing an element being target value in the classification task, i.e. identifier of a desired class.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class ImageValue(ElementType):
    """
        Type representing an element/value of a single image channel,
        e.g. a single element (R) of RGB image.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class NormalizedImageValue(ImageValue):
    """
        Type representing an element/value of a single image channel normalized to <0-1> range,
        e.g. a single element (R) of normalized RGB image.
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class ImageFeatureValue(ImageValue):
    """Type representing an element (single value) of a (image) feature maps."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class StringType(ElementType):
    """Element type representing a single string"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class StringLabel(StringType):
    """
        Type representing an label being a string with class name (e.g. the "hamster" class in CIFAR100).
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class BoolType(ElementType):
    """Element type representing a single integer"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class IntType(ElementType):
    """Element type representing a single integer"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class FloatType(ElementType):
    """Element type representing a single float"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""
        pass


class TokenIndex(IntType):
    """Type representing an element being index of a token in some kind of a vocabulary."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class Length(IntType):
    """Type representing an element storing a "length" (e.g. length of a list)."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class ProbabilityDistributionSamplesType(ElementType):
    """Element to represent tensors that meant to be sampled from a valid probability distribution
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class NormalDistributionSamplesType(ProbabilityDistributionSamplesType):
    """Element to represent tensors that meant to be sampled from a valid normal distribution
    """

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class SequenceToSequenceAlignmentType(ElementType):
    """Class to represent the alignment from seq-to-seq attention outputs. Generally a mapping from endcoder time steps
    to decoder time steps."""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class NormalDistributionMeanType(ElementType):
    """Element to represent the mean of a normal distribution"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class NormalDistributionLogVarianceType(ElementType):
    """Element to represent the log variance of a normal distribution"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class TokenDurationType(ElementType):
    """Element for representing the duration of a token"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class TokenLogDurationType(ElementType):
    """Element for representing the log-duration of a token"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass


class LogDeterminantType(ElementType):
    """Element for representing log determinants usually used in flow models"""

    def __init__(self):
        """Dummy init for TorchScript compatibility"""

    pass
