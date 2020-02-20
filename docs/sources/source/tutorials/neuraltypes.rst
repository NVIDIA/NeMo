Neural Types
============

Basics
~~~~~~

All input and output ports of every neural module in NeMo are typed.
The type system's goal is check compatibility of connected input/output port pairs.
The type system's constraints are checked when the user connects modules with each other and before any training or
inference is started.

Neural Types are implemented with the Python class :class:`NeuralType<nemo.core.neural_types.NeuralType>` and helper
classes derived from :class:`ElementType<nemo.core.neural_types.ElementType>`, :class:`AxisType<nemo.core
.neural_types.AxisType>` and :class:`AxisKindAbstract<nemo.core.neural_types.AxisKindAbstract>`.

**A Neural Type contains two categories of information:**

* **axes** - represents what varying a particular axis means (e.g. batch, time, etc.)
* **elements_type** - represents the semantics and properties of what is stored inside the activations (audio signal,text embedding, logits, etc.)


To instantiate a NeuralType you need to pass it the following arguments: `axes: Optional[Tuple] = None,
elements_type: ElementType = VoidType(), optional=False`. Typically, the only place where you need to instantiate
:class:`NeuralType<nemo.core.neural_types.NeuralType>` objects are inside your module's `input_ports` and
`output_ports` properties.


Consider an example below. It represents an (audio) data layer output ports, used in Speech recognition collection.

.. code-block:: python

        {
            'audio_signal': NeuralType(axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=False),
                                             AxisType(kind=AxisKind.Time, size=None, is_list=False)),
                                       elements_type=AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(axes=tuple(AxisType(kind=AxisKind.Batch, size=None, is_list=False)),
                                       elements_type=LengthsType()),
            'transcripts': NeuralType(axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=False),
                                             AxisType(kind=AxisKind.Time, size=None, is_list=False)),
                                      elements_type=LabelsType()),
            'transcript_length': NeuralType(axes=tuple(AxisType(kind=AxisKind.Batch, size=None, is_list=False)),
                                            elements_type=LengthsType()),
        }

A less verbose version of exactly the same output ports looks like this:

.. code-block:: python

        {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }



Neural type comparison
~~~~~~~~~~~~~~~~~~~~~~

Two :class:`NeuralType<nemo.core.neural_types.NeuralType>` objects are compared using ``.compare`` method.
The result is from the :class:`NeuralTypeComparisonResult<nemo.core.neural_types.NeuralTypeComparisonResult>`:

.. code-block:: python

    class NeuralTypeComparisonResult(Enum):
        """The result of comparing two neural type objects for compatibility.
        When comparing A.compare_to(B):"""

        SAME = 0
        LESS = 1  # A is B
        GREATER = 2  # B is A
        DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
        TRANSPOSE_SAME = 4  # A transpose and/or converting between lists and tensors will make them same
        CONTAINER_SIZE_MISMATCH = 5  # A and B contain different number of elements
        INCOMPATIBLE = 6  # A and B are incompatible
        SAME_TYPE_INCOMPATIBLE_PARAMS = 7  # A and B are of the same type but parametrized differently


Special cases
~~~~~~~~~~~~~

* **Void** element types. Sometimes, it is necessary to have a functionality similar to "void*" in C/C++. That, is if we still want to enforce order and axes' semantics but should be able to accept elements of any type. This can be achieved by using an instance of :class:`VoidType<nemo.core.neural_types.VoidType>` as ``elements_type`` argument.
* **Big void** this type will effectively disable any type checks. This is how to create such type: ``NeuralType()``. The result of its comparison to any other type will always be SAME.
* **AxisKind.Any** this axis kind is used to represent any axis. This is useful, for example, in losses where a specific loss module can be used in difference applications and therefore with different axis kinds

Inheritance
~~~~~~~~~~~

Type inheritance is a very powerful tool in programming. NeMo's neural types support inheritance. Consider the
following example below.

**Example.** We want to represent the following. A module's A output (out1) produces mel-spectrogram
signal, while module's B output produces mffc-spectrogram. We also want to a thrid module C which can perform data
augmentation with any kind of spectrogram. With NeMo neural types representing this semantics is easy:

.. code-block:: python

    input = NeuralType(('B', 'D', 'T'), SpectrogramType())
    out1 = NeuralType(('B', 'D', 'T'), MelSpectrogramType())
    out2 = NeuralType(('B', 'D', 'T'), MFCCSpectrogramType())

    # then the following comparison results will be generated
    input.compare(out1) == SAME
    input.compare(out2) == SAME
    out1.compare(input) == INCOMPATIBLE
    out2.compare(out1) == INCOMPATIBLE

This happens because both ``MelSpectrogramType`` and ``MFCCSpectrogramType`` inherit from ``SpectrogramType`` class.
Notice, that mfcc and mel spectrograms aren't interchangable, which is why ``out1.compare(input) == INCOMPATIBLE``

Advanced usage
~~~~~~~~~~~~~~

**Extending with user-defined types.** If you need to add your own element types, create a new class inheriting from
:class:`ElementType<nemo.core.neural_types.ElementType>`. Instead of using built-in axes kinds from
:class:`AxisKind<nemo.core.neural_types.AxisKind>`, you can define your own
by creating a new Python enum which should inherit from :class:`AxisKindAbstract<nemo.core.neural_types.AxisKindAbstract>`.

**Lists**. Sometimes module's input or output should be a list (possibly nested) of Tensors. NeMo's
:class:`AxisType<nemo.core.neural_types.AxisType>` class accepts ``is_list`` argument which could be set to True.
Consider the example below:

.. code-block:: python

        T1 = NeuralType(
            axes=(
                AxisType(kind=AxisKind.Batch, size=None, is_list=True),
                AxisType(kind=AxisKind.Time, size=None, is_list=True),
                AxisType(kind=AxisKind.Dimension, size=32, is_list=False),
                AxisType(kind=AxisKind.Dimension, size=128, is_list=False),
                AxisType(kind=AxisKind.Dimension, size=256, is_list=False),
            ),
            elements_type=ChannelType(),
        )

In this example, first two axes are lists. That is the object are list of lists of rank 3 tensors with dimensions
(32x128x256). Note that all list axes must come before any tensor axis.

.. tip::
    We strongly recommend this to be avoided, if possible, and tensors used instead (perhaps) with padding.


**Named tuples (structures).** To represent struct-like objects, for example, bounding boxes in computer vision, use
the following syntax:

.. code-block:: python

        class BoundingBox(ElementType):
            def __str__(self):
                return "bounding box from detection model"
            def fields(self):
                return ("X", "Y", "W", "H")
        # ALSO ADD new, user-defined, axis kind
        class AxisKind2(AxisKindAbstract):
            Image = 0
        T1 = NeuralType(elements_type=BoundingBox(),
                        axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=True),
                              AxisType(kind=AxisKind2.Image, size=None, is_list=True)))

In the example above, we create a special "element type" class for BoundingBox which stores exactly 4 values.
We also, add our own axis kind (Image). So the final Neural Type (T1) represents lists (for batch) of lists (for
image) of bounding boxes. Under the hood it should be list(lists(4x1 tensors)).


**Neural Types help us to debug models**

There is a large class of errors, which will not produce runtime or compile exception. For example:

(1) "Rank matches but semantics doesn't".

For example, Module A produces data in the format [Batch, Time, Dim] whereas Module B  expects format [Time, Batch, Dim]. A simple axes transpose will solve this error.

(2) "Concatenating wrong dimensions".

For example, module should concatenate (add) two input tensors X and Y along dimension 0. But tensor X is in format [B, T, D] while tensor Y=[T, B, D] and concat . .

(3) "Dimensionality mismatch"

A module expects image of size 224x224 but gets 256x256. The type comparison will result in ``NeuralTypeComparisonResult.DIM_INCOMPATIBLE`` .



