模块配置
=========

神经模块的配置可以从 YAML 文件导入，也可以导出到 YAML 文件 \
一个模块配置文件存储了创建一个实例所需要的所有参数。

.. note::
    对于可训练的神经模块，`配置`相对于 checkpoint 是起到了互补的作用。
    配置文件包含了参数 (比如: 层的数量, 隐藏层的大小等), \
    而 checkpoint 包含了实际模块的权重


导出配置文件
---------------

在下面的例子中，我们再次训练一个模型来学习 y=sin(x) 的泰勒系数 \
但是，我们进一步的扩展了这个例子，展示了如果导出模块的配置，并写入到一个 YAML 文件中 \
再用相同的参数创建第二个实例

我们首先创建 :class:`NeuralFactory` 对象，从原始例子中初始化这个模块:

.. literalinclude:: ../../../../examples/start_here/module_configuration.py
   :language: python
   :lines: 25-35

现在我们可以导出任何一个已有模块的配置，调用 :meth:`export_to_config()`, 例如 \
我们可以导出 :class:`TaylorNet` 的配置，通过调用:

.. literalinclude:: ../../../../examples/start_here/module_configuration.py
   :language: python
   :lines: 38

导入配置
---------

有个类似的函数 :meth:`import_from_config()` 负责加载配置文件:

.. literalinclude:: ../../../../examples/start_here/module_configuration.py
   :language: python
   :lines: 41

.. note::
    :meth:`import_from_config()` 函数事实上是创建了在配置中的这个类的一个新的实例 \
    需要注意的是这两个类不贡献任何可训练的权重。 \
    NeMo 给权重连接(weight tying)提供了另一套机制。

现在我们可以像用其它模块那样用导入的模块 \
例如，我们可以构建一个图，用 NeMo trainer 来训练:

.. literalinclude:: ../../../../examples/start_here/module_configuration.py
   :language: python
   :lines: 43-


.. include:: module_custom_configuration.rst


.. note::
    上面(以及其它许多的)的例子可以在 `nemo/examples` 文件夹下找到
