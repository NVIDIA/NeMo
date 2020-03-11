自定义配置
-----------


一个一般的配置导出应该可以让我们用基于原始数据类型(string, int, float)的参数 \
或者用 list/dict 嵌套的原始数据类型。

如果想拓展这个功能，用其它的自定义类型，用户需要为自己的模块类，重载 \
方法 :meth:`export_to_config()` 和  :meth:`import_from_config()`  \
下面这个教程解释了我们该怎么做。


下面的例子中，我们从 :class:`TaylorNet` (在前面的例子中有使用过这个类) 中得到我们的类 \
然后用这些方法来拓展它。首先我们定义一个简单的类 :class:`Status` enum:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 33-35

现在让我们定义 :class:`CustomTaylorNet` 神经模块类:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 38-43


为了能处理好 :class:`Status` enum 的导出功能，我们必须实现自定义函数 \
:meth:`export_to_config()`:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 45-76


注意配置实际上是一个字典，包含了两个部分:

 * ``header`` (存储类的说明, NeMo 版本, NeMo 集合名称等) 以及
 * ``init_params`` 存储初始化对象所需要的参数

这些参数存在保护域 ``self._init_params`` 中，它的基类是 :class:`NeuralModule` 类。
确保用户不能直接访问和使用它们。 

类似地，我们必须重载方法 :meth:`import_from_config()` :

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 79-119

请注意，基类 :class:`NeuralModule` 提供了一些保护方法供我们使用, \
其中，最重要的是:

 * :meth:`_create_config_header()` 生成合适的 header, 以及 \
 * :meth:`_validate_config_file()` 验证加载的配置文件 (检查 header 内容)。


.. note::
    再强调一下 :meth:`import_from_config()` 是类的方法，实际上返回 \
    一个新的对象实例 - 在这个例子中就是 :class:`CustomTaylorNet` 类型。


现在我们可以简单的构建一个实例，并且导出它的配置，通过调用:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 128-129,134-135

通过加载这个配置，初始化第二个实例:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 137-139

从结果中我们可以看到新的对象把状态都设置成了原来那个对象的值:

.. code-block:: bash

    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:74] Configuration of module 3ec99d30-baba-4e4c-a62b-e91268762864 (CustomTaylorNet) exported to /tmp/custom_taylor_net.yml
    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:41] Status: Status.error
    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:114] Instantiated a new Neural Module of type `CustomTaylorNet` using configuration loaded from the `/tmp/custom_taylor_net.yml` file
