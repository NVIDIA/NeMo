Neural Types
============

Neural Types are used to check input tensors to make sure that two neural modules are compatible, and catch
semantic and dimensionality errors.

Neural Types are implemented by :class:`NeuralType<nemo.core.neural_types.NeuralType>` class which is a mapping from Tensor's axis to :class:`AxisType<nemo.core.neural_types.AxisType>`.

:class:`AxisType<nemo.core.neural_types.AxisType>` contains following information per axis:

* Semantic Tag, which must inherit from :class:`BaseTag<nemo.core.neural_types.BaseTag>`, for example: :class:`BatchTag<nemo.core.neural_types.BatchTag>`, :class:`ChannelTag<nemo.core.neural_types.ChannelTag>`, :class:`TimeTag<nemo.core.neural_types.TimeTag>`, etc. These tags can be related via `is-a` inheritance.
* Dimension: unsigned integer
* Descriptor: string


To instantiate a NeuralType you should pass it a dictionary (axis2type) which will map axis to it's AxisType.
For example, a ResNet18 input and output ports can be described as:

.. code-block:: python

    input_ports = {"x": NeuralType({0: AxisType(BatchTag),
                                    1: AxisType(ChannelTag),
                                    2: AxisType(HeightTag, 224),
                                    3: AxisType(WidthTag, 224)})}
    output_ports = {"output": NeuralType({
                                    0: AxisType(BatchTag),
                                    1: AxisType(ChannelTag)})}



**Neural type comparison**

Two :class:`NeuralType<nemo.core.neural_types.NeuralType>` objects can be compared using ``.compare`` method.
The result is:

.. code-block:: python

    class NeuralTypeComparisonResult(Enum):
      """The result of comparing two neural type objects for compatibility.
      When comparing A.compare_to(B):"""
      SAME = 0
      LESS = 1  # A is B
      GREATER = 2  # B is A
      DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
      TRANSPOSE_SAME = 4 # A transpose will make them same
      INCOMPATIBLE = 5  # A and B are incompatible. Can't fix incompatibility automatically


**Special cases**

* *Non-tensor* objects should be denoted as ``NeuralType(None)``
* *Optional*: input is as optional, if input is provided the type compatibility will be checked
* *Root* type is denoted by ``NeuralType({})``: A port of ``NeuralType({})`` type must accept NmTensors of any NeuralType:

.. code-block:: python

    root_type = NeuralType({})
    root_type.compare(any_other_neural_type) == NeuralTypeComparisonResult.SAME

See "nemo/tests/test_neural_types.py" for more examples.


**Neural Types help us to debug models**

There is a large class of errors, which will not produce runtime or compile exception. For example:

(1) "Rank matches but semantics doesn't".

For example, Module A produces data in the format [Batch, Time, Dim] whereas Module B  expects format [Time, Batch, Dim]. A simple axes transpose will solve this error.

(2) "Concatenating wrong dimensions".

For example, module should concatenate (add) two input tensors X and Y along dimension 0. But tensor X is in format [B, T, D] while tensor Y=[T, B, D] and concat . .

(3) "Dimensionality mismatch"

A module expects image of size 224x224 but gets 256x256. The type comparison will result in ``NeuralTypeComparisonResult.DIM_INCOMPATIBLE`` .

.. note::
    This type mechanism is represented by Python inheritance. That is, :class:`NmTensor<nemo.core.neural_types.NmTensor>` class inherits from :class:`NeuralType<nemo.core.neural_types.NeuralType>` class.

