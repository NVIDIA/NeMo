神经类型
============

神经类型是用来检查输入张量，确保两个神经模块是兼容的，并且捕捉语义和维度上的错误。

神经类型在 :class:`NeuralType<nemo.core.neural_types.NeuralType>` 类中实现，它把张量的轴映射到 :class:`AxisType<nemo.core.neural_types.AxisType>`。

:class:`AxisType<nemo.core.neural_types.AxisType>` 每个轴包含下列信息：

* 语义标签（Semantic Tag）, 必须继承 :class:`BaseTag<nemo.core.neural_types.BaseTag>`类，比如 :class:`BatchTag<nemo.core.neural_types.BatchTag>`, :class:`ChannelTag<nemo.core.neural_types.ChannelTag>`, :class:`TimeTag<nemo.core.neural_types.TimeTag>` 等。这些标签是 `is-a` 的继承关系。
* 维度（Dimension）: 无符号整形
* 描述符（Descriptor）: 字符串


初始化神经类型，你应该给它传递一个字典（轴到类型），把轴映射到它的AxisType。
比如，ResNet18 的输入和输出端口可以这么描述：

.. code-block:: python

    input_ports = {"x": NeuralType({0: AxisType(BatchTag),
                                    1: AxisType(ChannelTag),
                                    2: AxisType(HeightTag, 224),
                                    3: AxisType(WidthTag, 224)})}
    output_ports = {"output": NeuralType({
                                    0: AxisType(BatchTag),
                                    1: AxisType(ChannelTag)})}



**神经类型比较**

两个 :class:`NeuralType<nemo.core.neural_types.NeuralType>` 对象可以通过 ``.compare`` 方法来进行比较。
结果是:

.. code-block:: python

    class NeuralTypeComparisonResult(Enum):
      """比较两个神经类型兼容性的结果
      A.compare_to(B):"""
      SAME = 0
      LESS = 1  # A 是 B
      GREATER = 2  # B 是 A
      DIM_INCOMPATIBLE = 3  # 重新调整连接器也许可以修复不兼容
      TRANSPOSE_SAME = 4 # 把 A 转置可以使它们相同
      INCOMPATIBLE = 5  # A 和 B 不兼容。不能自动修复不兼容


**特殊例子**

* *Non-tensor* 对象应该用 ``NeuralType(None)`` 表示。
* *Optional*: 输入是可选的，如果提供了类型输入，那么会自动做类型检测
* *Root* 类型可以用 ``NeuralType({})`` 表示: ``NeuralType({})`` 类型的端口必须可以接收任意的神经类型的神经模块张量（NmTensors）：

.. code-block:: python

    root_type = NeuralType({})
    root_type.compare(any_other_neural_type) == NeuralTypeComparisonResult.SAME

参考 "nemo/tests/test_neural_types.py" 中更多的例子。


**神经类型帮助我们调试程序**

有许多的错误类型在运行和编译的时候不会报错，比如：

(1) "Rank matches but semantics doesn't".

例如，模块 A 的数据格式是 [Batch, Time, Dim]，但是模块B期望的格式是 [Time, Batch, Dim]。简单的轴转置就可以解决这个错误。

(2) "Concatenating wrong dimensions".

例如, 模块应该根据 0 号维度合并（加）两个输入张量 X 和 Y。但是张量 X 格式是 [B, T, D]，但是张量 Y 格式是 [T, B, D] 然后做合并。

(3) "Dimensionality mismatch"

一个模块期望图片尺寸是 224x224 但是得到的是 256x256。这种类型比较会导致 ``NeuralTypeComparisonResult.DIM_INCOMPATIBLE``。

.. note::
    这个类型机制是由 Python 继承表示的。也就是说 :class:`NmTensor<nemo.core.neural_types.NmTensor>` 类继承自 :class:`NeuralType<nemo.core.neural_types.NeuralType>` 类。

