神经类型
============

基础
~~~~~~

每个神经模块的输入和输出端口都是有类型的。
类型系统的目标是要检查相连输入/输出端口对之间的兼容性。
当用户连接各个模块以及在训练和推理开始之前，类型系统的约束限制都会被检查。

神经类型 (Neural Types) 在 Python 类 :class:`NeuralType<nemo.core.neural_types.NeuralType>` 中实现，帮助类
由 :class:`ElementType<nemo.core.neural_types.ElementType>`, :class:`AxisType<nemo.core
.neural_types.AxisType>` 和 :class:`AxisKindAbstract<nemo.core.neural_types.AxisKindAbstract>` 这几个类得到。

**一个神经类型包含两类信息**

* **axes** - 表示特定轴的含义 (e.g. batch, time, 等)
* **elements_type** - 表示存在里面的激活元的语义和属性 (audio signal,text embedding, logits, 等)


如果想初始化一个NeuralType, 你需要传递给它下面的参数: `axes: Optional[Tuple] = None,
elements_type: ElementType = VoidType(), optional=False`. 通常，初始化
:class:`NeuralType<nemo.core.neural_types.NeuralType>` 对象的地方是在模块里面的 `input_ports` 和
`output_ports` 属性中。


考虑下面的这个例子。它表示了一个在语音识别集合中用到的(音频) 数据层的输出端口。

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

一个具有一样输出端口的更加精简的版本:

.. code-block:: python

        {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }



神经类型比较
~~~~~~~~~~~~~~~~~~~~~~

两个 :class:`NeuralType<nemo.core.neural_types.NeuralType>` 对象可以用 ``.compare`` 方法来进行比较。
比较的结果来自 :class:`NeuralTypeComparisonResult<nemo.core.neural_types.NeuralTypeComparisonResult>`:

.. code-block:: python

    class NeuralTypeComparisonResult(Enum):
        """比较两个神经类型对象兼容性的比较结果，
        使用 A.compare_to(B):"""

        SAME = 0
        LESS = 1  # A 是 B
        GREATER = 2  # B 是 A
        DIM_INCOMPATIBLE = 3  # 调整连接的大小 (resize) 也许可以修复不兼容性
        TRANSPOSE_SAME = 4  # 转置以及/或者在 lists 和 tensors 之间的转换可以让它们一致
        CONTAINER_SIZE_MISMATCH = 5  # A 和 B 包含不同数量的元素
        INCOMPATIBLE = 6  # A 和 B 不兼容
        SAME_TYPE_INCOMPATIBLE_PARAMS = 7  # A 和 B 相同类型但参数化不同


特殊例子
~~~~~~~~~~~~~

* **Void** 元素类型。有时候，有个像  C/C++ 中 "void*" 的功能还挺有必要的。也就是说，我们想强制顺序(order)和轴的语义但是又要求能接受任何类型的元素。我们可以用 :class:`VoidType<nemo.core.neural_types.VoidType>` 实例作为 ``elements_type`` 。
* **Big void** 这种类型会取禁用掉所有的类型检查。可以这样创建这个类型: ``NeuralType()``. 它和其它类型的比较结果永远都是 SAME。
* **AxisKind.Any** 这个轴类型(kind)用来表示任意的轴类型。这个很管用，比如，在损失函数中，一个特定的损失函数模块可以用在不同的应用中，表示不同的轴类型。

继承
~~~~~~~~~~~

类型继承在编程中是非常强大的工具。 NeMo 的神经类型支持继承。考虑
下面这个例子。

**例子.** 我们想要表示: A 模块的 A 输出 (out1) 产生梅尔谱(mel-spectrogram)
信号, 而模块 B 输出产生 mffc 频谱。我们也想要一个模块 C 可以对任意频谱做数据增强
用 NeMo 的神经类型表示这种语义就很容易:

.. code-block:: python

    input = NeuralType(('B', 'D', 'T'), SpectrogramType())
    out1 = NeuralType(('B', 'D', 'T'), MelSpectrogramType())
    out2 = NeuralType(('B', 'D', 'T'), MFCCSpectrogramType())

    # 会生成下面的结果
    input.compare(out1) == SAME
    input.compare(out2) == SAME
    out1.compare(input) == INCOMPATIBLE
    out2.compare(out1) == INCOMPATIBLE

之所以会这样是因为 ``MelSpectrogramType`` 和 ``MFCCSpectrogramType`` 都继承自 ``SpectrogramType`` 类。
注意, mfcc 和 mel 频谱是不能互换的，这就是为什么 ``out1.compare(input) == INCOMPATIBLE``

高级用法
~~~~~~~~~~~~~~

**使用用户定义的类型** 如果你相加自己的元素类型, 创建一个新的继承类
:class:`ElementType<nemo.core.neural_types.ElementType>` 的类。除了使用内置的轴类型
:class:`AxisKind<nemo.core.neural_types.AxisKind>`, 你可以定义自己的轴类型
创建一个新的 Python enum, 继承 :class:`AxisKindAbstract<nemo.core.neural_types.AxisKindAbstract>`.

**列表(Lists)**. 有时候模块的输入和输出应该是一个 List 的(也有可能是嵌套的)张量。 NeMo 的
:class:`AxisType<nemo.core.neural_types.AxisType>` 类接受 ``is_list`` 参数，它可以设置为 True。
考虑下面的例子:

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

这个例子中，前两个轴是 list。这个对象的 list 中的 list 中的元素秩为3的张量，张量维度为(32x128x256).
注意 list 的轴必须在其它张量轴的前面。

.. tip::
    我们强烈建议避免这么做。可能的话还是用张量带 padding 的方式来做。


**命名元组(Named tuples) (数据结构).** 为了能够表示结构化的对象, 例如：在计算机视觉中的边界框(bounding box), 
可以用下面的语句:

.. code-block:: python

        class BoundingBox(ElementType):
            def __str__(self):
                return "bounding box from detection model"
            def fields(self):
                return ("X", "Y", "W", "H")
        # 加新的用户定义的轴类型
        class AxisKind2(AxisKindAbstract):
            Image = 0
        T1 = NeuralType(elements_type=BoundingBox(),
                        axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=True),
                              AxisType(kind=AxisKind2.Image, size=None, is_list=True)))

在上面的例子中, 我们给边界框构建了一个特别的 "element type" 类，包含了4个值。
我们也加了自己的轴类型(Image). 所以最后的神经类型(T1) 表示的是 lists(batch) 的 lists (
image) 的边界框。就是说，它是 list(lists(4x1 张量))。


**神经类型帮助我们调试模型**

有一个很大的错误类, 在运行和编译的时候不会报错。比如:

(1) "Rank matches but semantics doesn't".

比如，模块 A 产生的数据格式是 [Batch, Time, Dim] 而模块 B 期望的格式是 [Time, Batch, Dim]。简单的轴转置就可以解决这个错。

(2) "Concatenating wrong dimensions".

例如, 模块应该沿着维度 0 合并(相加)两个输入张量 X 和 Y。但是张量 X 格式为 [B, T, D] 而张量 Y=[T, B, D] 然后合并. .

(3) "Dimensionality mismatch"

一个模块想要一张大小为 224x224 的图片，但得到的是 256x256。类型比较的结果是 ``NeuralTypeComparisonResult.DIM_INCOMPATIBLE`` 。



