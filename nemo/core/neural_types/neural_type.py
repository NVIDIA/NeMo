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
    'NeuralType',
    'NmTensor',
    'NeuralTypeError',
    'NeuralPortNameMismatchError',
    'NeuralPortNmTensorMismatchError',
    'CanNotInferResultNeuralType',
]
import uuid
from typing import Optional, Tuple

from nemo.core.neural_types.axes import AxisKind, AxisType
from nemo.core.neural_types.comparison import NeuralTypeComparisonResult
from nemo.core.neural_types.elements import *


class NeuralType(object):
    """This is the main class which would represent neural type concept.
    nmTensors derives from this. It is used to represent *the types* of inputs and outputs.
    Args:
        axes (Optional[Tuple]): a tuple of AxisTypes objects representing the semantics of what varying each axis means
            You can use a short, string-based form here. For example: ('B', 'C', 'H', 'W') would correspond to an NCHW
            format frequently used in computer vision. ('B', 'T', 'D') is frequently used for signal processing and
            means [batch, time, dimension/channel].
        elements_type (ElementType): an instance of ElementType class representing the semantics of what is stored
            inside the tensor. For example: logits (LogitsType), log probabilities (LogprobType), etc.
        optional (bool): By default, this is false. If set to True, it would means that input to the port of this
            type can be optional.
    """

    def __str__(self):

        if self.axes is not None:
            return f"axes: {self.axes}; " f" elements_type: {self.elements_type.__class__.__name__}"
        else:
            return f"axes: None; " f" elements_type: {self.elements_type.__class__.__name__}"

    def __init__(self, axes: Optional[Tuple] = None, elements_type: ElementType = VoidType(), optional=False):
        if not isinstance(elements_type, ElementType):
            raise ValueError(
                f"elements_type of NeuralType must be an instance of a class derived from ElementType."
                f"Did you pass a class instead?"
            )
        self.elements_type = elements_type
        if axes is not None:
            NeuralType.__check_sanity(axes)
            axes_list = []
            for axis in axes:
                if isinstance(axis, str):
                    axes_list.append(AxisType(AxisKind.from_str(axis), None))
                elif isinstance(axis, AxisType):
                    axes_list.append(axis)
                else:
                    raise ValueError(f"axis type must be either str or AxisType instance")
            self.axes = tuple(axes_list)
        else:
            self.axes = None
        self.optional = optional

    def compare(self, second) -> NeuralTypeComparisonResult:
        """Performs neural type comparison of self with second. When you chain two modules' inputs/outputs via
        __call__ method, this comparison will be called to ensure neural type compatibility."""
        # First, handle dimensionality
        axes_a = self.axes
        axes_b = second.axes

        # "Big void" type
        if isinstance(self.elements_type, VoidType) and self.axes is None:
            return NeuralTypeComparisonResult.SAME

        if self.axes is None:
            if second.axes is None:
                return self.elements_type.compare(second.elements_type)
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE

        dimensions_pass = NeuralType.__compare_axes(axes_a, axes_b)
        element_comparison_result = self.elements_type.compare(second.elements_type)

        # SAME DIMS
        if dimensions_pass == 0:
            return element_comparison_result
        # TRANSPOSE_SAME DIMS
        elif dimensions_pass == 1:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.TRANSPOSE_SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        # DIM_INCOMPATIBLE DIMS
        elif dimensions_pass == 2:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    @staticmethod
    def __check_sanity(axes):
        # check that list come before any tensor dimension
        are_strings = True
        for axis in axes:
            if not isinstance(axis, str):
                are_strings = False
            if isinstance(axis, str) and not are_strings:
                raise ValueError("Either use full class names or all strings")
        if are_strings:
            return
        checks_passed = True
        saw_tensor_dim = False
        for axis in axes:
            if not axis.is_list:
                saw_tensor_dim = True
            else:  # current axis is a list
                if saw_tensor_dim:  # which is preceded by tensor dim
                    checks_passed = False
        if not checks_passed:
            raise ValueError(
                "You have list dimension after Tensor dimension. All list dimensions must preceed Tensor dimensions"
            )

    @staticmethod
    def __compare_axes(axes_a, axes_b) -> int:
        """
        Compares axes_a and axes_b
        Args:
            axes_a: first axes tuple
            axes_b: second axes tuple

        Returns:
            0 - if they are exactly the same
            1 - if they are "TRANSPOSE_SAME"
            2 - if the are "DIM_INCOMPATIBLE"
            3 - if they are different
        """
        if axes_a is None and axes_b is None:
            return 0
        elif axes_a is None and axes_b is not None:
            return 3
        elif axes_a is not None and axes_b is None:
            return 3
        elif len(axes_a) != len(axes_b):
            return 3
        # After these ifs we know that len(axes_a) == len(axes_b)

        same = True
        kinds_a = dict()
        kinds_b = dict()
        for axis_a, axis_b in zip(axes_a, axes_b):
            kinds_a[axis_a.kind] = axis_a.size
            kinds_b[axis_b.kind] = axis_b.size
            if axis_a.kind == AxisKind.Any:
                same = True
            elif (
                axis_a.kind != axis_b.kind
                or axis_a.is_list != axis_b.is_list
                or (axis_a.size != axis_b.size and axis_a.size is not None)
            ):
                same = False
        if same:
            return 0
        else:
            # can be TRANSPOSE_SAME, DIM_INCOMPATIBLE
            if kinds_a.keys() == kinds_b.keys():
                for key, value in kinds_a.items():
                    if kinds_b[key] != value:
                        return 2
                return 1
            else:
                return 3


class NmTensor(NeuralType):
    """Class representing data which flows between NeuralModules' ports.
    It also has a type of NeuralType represented by inheriting from NeuralType
    object."""

    def __init__(self, producer, producer_args, name, ntype=None):
        """NmTensor constructor.

        Args:
          producer (NeuralModule): object which produced this
          producer_args (dict): a dictionary of port_name->NmTensor value
            of arguments which were sent to producer to create this
        """
        super(NmTensor, self).__init__(axes=ntype.axes, elements_type=ntype.elements_type, optional=ntype.optional)
        self._producer = producer
        self._producer_args = producer_args
        self._name = name
        self._uuid = str(uuid.uuid4())

    @property
    def producer(self):
        """
        Returns:
          NeuralModule object which produced this NmTensor.
        """
        return self._producer

    @property
    def producer_args(self):
        """
        Returns:
          a dictionary of port_name->NmTensor value
          of arguments which were sent to producer to create this object
        """
        return self._producer_args

    @property
    def name(self):
        """
        Returns:
          A NmTensor's name which should be equal to
          the NeuralModule's output port's name which created it
        """
        return self._name

    @property
    def unique_name(self):
        """Unique NMTensor name.
        It is composed of non-unique name (self.name) and uuid of NeuralModule
        which created this tensor.

        Returns:
          str: unique name
        """
        if self._producer is None:
            raise ValueError("This NmTensor does not have a unique name")
        return f"{self._name}~~~{self.producer}~~~{self._uuid}"


class NeuralTypeError(Exception):
    """Base class for neural type related exceptions."""

    pass


class NeuralPortNameMismatchError(NeuralTypeError):
    """Exception raised when neural module is called with incorrect port
    names."""

    def __init__(self, message):
        self.message = message


class NeuralPortNmTensorMismatchError(NeuralTypeError):
    """Exception raised when a port is fed with a NmTensor of incompatible
    type."""

    def __init__(self, message):
        self.message = message


class CanNotInferResultNeuralType(NeuralTypeError):
    """Exception raised when NeuralType of output can not be inferred."""

    def __init__(self, message):
        self.message = message
