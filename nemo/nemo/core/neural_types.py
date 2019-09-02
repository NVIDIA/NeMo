# Copyright (c) 2019 NVIDIA Corporation
"""This module contains Tags, AxisTypes, NeuralTypes and NmTensors.
Every NmTensor is of a particular Neural Type.
Neural Modules' input and output ports are also of Neural Type.

An exception will be raised when a NmTensor and input port where it goes are
of incompatible types.
"""
from enum import Enum
import uuid


class BaseTag(object):
    """Base Neural Tag. All Tags should inherit from this."""

    def __str__(self):
        return "base"


class BatchTag(BaseTag):
    """Tag for batch dimension."""

    def __str__(self):
        return "batch"


class TimeTag(BaseTag):
    """Tag for time dimension."""

    def __str__(self):
        return "time"


class ProcessedTimeTag(TimeTag):
    """Tag for processed time dimension.
    For example: after pre-processing, or augmentation."""

    def __str__(self):
        return "processed_time"


class ChannelTag(BaseTag):
    """Tag for channel dimension."""

    def __str__(self):
        return "channel"


class SpectrogramSignalTag(ChannelTag):
    """Tag for spectrogram signal dimension."""

    def __str__(self):
        return "spectrogram_signal"


class EncodedRepresentationTag(ChannelTag):
    """Tag for encoded representation. This should be used to
    denote encoders' outputs."""

    def __str__(self):
        return "encoded_representation"


class ClassTag(BaseTag):
    """Tag for class dimension.
    For example, number of classes in classification problem,
    vocabuary size or num of charachters for ASR."""

    def __str__(self):
        return "channel"


class WidthTag(BaseTag):
    """Tag for width dimension."""

    def __str__(self):
        return "width"


class HeightTag(BaseTag):
    """Tag for width dimension."""

    def __str__(self):
        return "height"


class NeuralTypeComparisonResult(Enum):
    """The result of comparing two neural type objects for compatibility.
    When comparing A.compare_to(B):"""

    SAME = 0
    LESS = 1  # A is B
    GREATER = 2  # B is A
    DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
    TRANSPOSE_SAME = 4  # A transpose will make them same
    INCOMPATIBLE = (
        5
    )  # A and B are incompatible. Can't fix incompatibility automatically


class AxisType(object):
    """Every tensor's axis has semantics, dimension and descriptor.
    It's semantics is a Neural Tag (inherited from BaseTag)
    dimension (dim) is (optional) int and descriptor is (optional) string"""

    def __init__(self, semantics, dim: int = None,
                 descriptor: str = None):
        self._semantics = semantics
        self._dim = dim
        self._descriptor = descriptor

    def __eq__(self, other):
        return (
                self.semantics == other.semantics
                and self.dim == other.dim
                and self.descriptor == other.descriptor
        )

    def __str__(self):
        return "{0}:{1}:{2}".format(self.semantics, self.dim, self.descriptor)

    def __hash__(self):
        return hash(self.__str__())

    def compare_to(self, other):
        """
        Compares current AxisType object to other

        Args:
          other (AxisType): other AxisType object to compare with

        Returns:
          Results of a comparison (NeuralTypeComparisonResult)
        """
        if (
                self.dim is None or self.dim == other.dim
        ) and self.descriptor == other.descriptor:
            if self.semantics == other.semantics:
                return NeuralTypeComparisonResult.SAME
            elif issubclass(self.semantics, other.semantics):
                return NeuralTypeComparisonResult.LESS
            elif issubclass(other.semantics, self.semantics):
                return NeuralTypeComparisonResult.GREATER
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        elif self.descriptor == other.descriptor and self.semantics == \
                other.semantics:
            return NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    @property
    def semantics(self):
        return self._semantics

    @property
    def dim(self):
        return self._dim

    @property
    def descriptor(self):
        return self._descriptor


class NeuralType(object):
    """Neural Type: a type for NmTensor.

    Note: This type mechanism is represented by Python inheritance. That is,
    NmTensor
    class inherits from NeuralType class.

    A Neural Type is a mapping from Tensor's axis number to it's type (
    AxisType).

    To instantiate a NeuralType you should pass it a dictionary (axis2type)
    which
    will map axis to it's AxisType. You can also pass optional argument when
    describing input ports.

    For example, a ResNet18 input can be described as:

    .. code-block:: python

      NeuralType({0: AxisType(BatchTag, None, None),
                  1: AxisType(ChannelTag, None, None),
                  2: AxisType(HeightTag, 224, None),
                  3: AxisType(WidthTag, 224, None)})

    Special cases:
      - non-tensor objects should be denoted as NeuralType(None)
      - root type is denoted by NeuralType({}). A port of NeuralType({}) must

      accept NmTensors of any NeuralType. More specifically:
      root_type = NeuralType({})
      root_type.compare(any_other_neural_type) ==
      NeuralTypeComparisonResult.SAME


    See "nemo/tests/test_neural_types.py" for more examples.

    """

    # def __init__(self, axis2type=None):
    def __init__(self, axis2type={}, optional=False):
        """
        Constructor
        Args:
          axis2type: mapping axises to it's AxisType
          optional: (default: False). If this port is optional
        """
        self._axis2type = axis2type
        self._optional = optional

    def __str__(self):
        if self._axis2type is None:
            return "(Optional) " if self._optional else "" + "non-tensor " \
                                                             "object"
        elif len(self._axis2type) == 0:
            return "(Optional) " if self._optional else "" + "Root NeuralType"
        return (
            "(Optional)"
            if self._optional
            else ""
                 + "\n".join(["{0}->{1}".format(axis, tag) for axis, tag in
                             self._axis2type.items()])
        )

    def compare(self, n_type2) -> NeuralTypeComparisonResult:
        """Compares if current object's NeuralType semantics is compatible
        with n_type2

        Args:
          n_type2 (NeuralType): a type to compare with

        Returns:
          Results of a comparison (NeuralTypeComparisonResult)
        """
        # self is a root type
        if self.axis2type is not None and len(self.axis2type) == 0:
            return NeuralTypeComparisonResult.SAME
        # n_type2 is root type but self is not
        elif n_type2.axis2type is not None and len(n_type2.axis2type) == 0:
            return NeuralTypeComparisonResult.INCOMPATIBLE
        # one is None while other is not
        elif self._axis2type is None and n_type2._axis2type is not None:
            return NeuralTypeComparisonResult.INCOMPATIBLE
        elif self._axis2type is not None and n_type2._axis2type is None:
            return NeuralTypeComparisonResult.INCOMPATIBLE
        # same neural type
        elif self._axis2type == n_type2._axis2type:
            return NeuralTypeComparisonResult.SAME
        # same set of keys and set of values => TRANSPOSE_SAME
        elif set(self._axis2type.keys()) == set(
                n_type2._axis2type.keys()) and set(
            self._axis2type.values()
        ) == set(n_type2._axis2type.values()):
            return NeuralTypeComparisonResult.TRANSPOSE_SAME

        elif set(self._axis2type.keys()) == set(n_type2._axis2type.keys()):
            # comparison_result = 1
            comparison_result = 0
            for key in self._axis2type.keys():
                comparison_result = max(
                    self._axis2type[key].compare_to(
                        n_type2._axis2type[key]).value,
                    comparison_result,
                )
            return NeuralTypeComparisonResult(comparison_result)
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    @property
    def axis2type(self):
        return self._axis2type


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
        super(NmTensor, self).__init__(axis2type=ntype._axis2type)
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
