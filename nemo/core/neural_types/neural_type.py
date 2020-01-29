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
    'NeuralPortNmTensorMismatchError',
    'CanNotInferResultNeuralType',
]
import uuid
from typing import Tuple
from .comparison import NeuralTypeComparisonResult
from .axes import AxisType, AxisKind
from .elements import *


class NeuralType(object):
    """This is the main class which would represent neural type concept.
    nmTensors derives from this. It is used to represent *the types* of inputs and outputs."""

    def __init__(self, elements_type: ElementType, axes: Tuple, optional=False):
        self.__check_sanity(axes)
        self.elements_type = elements_type
        axes_list = []
        for axis in axes:
            if isinstance(axis, str):
                axes_list.append(AxisType(AxisKind.from_str(axis), None))
            elif isinstance(axis, AxisType):
                axes_list.append(axis)
            else:
                raise ValueError(f"axis type must be either str or AxisType instance")
        self.axes_tuple = tuple(axes_list)
        self.optional = optional

    def compare(self, second) -> NeuralTypeComparisonResult:
        # First, handle dimensionality
        axes_a = self.axes_tuple
        axes_b = second.axes_tuple

        kinds_a = dict()
        kinds_b = dict()

        dimensions_pass = True
        for axis_a, axis_b in zip(axes_a, axes_b):
            kinds_a[axis_a.kind] = axis_a.size
            kinds_b[axis_b.kind] = axis_b.size
            if axis_a.kind != axis_b.kind or axis_a.is_list != axis_b.is_list:
                dimensions_pass = False

        if kinds_a.keys() != kinds_b.keys():
            return NeuralTypeComparisonResult.INCOMPATIBLE
        for kind, size in kinds_a.items():
            if size != kinds_b[kind]:
                return NeuralTypeComparisonResult.DIM_INCOMPATIBLE

        element_comparison_result = self.elements_type.compare(second.elements_type)
        if dimensions_pass:
            return element_comparison_result
        elif element_comparison_result == NeuralTypeComparisonResult.SAME:
            return NeuralTypeComparisonResult.TRANSPOSE_SAME
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    def __check_sanity(self, axes):
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
        super(NmTensor, self).__init__(elements_type=ntype.elemts_type, axes=ntype.axes, optional=ntype.optional)
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
