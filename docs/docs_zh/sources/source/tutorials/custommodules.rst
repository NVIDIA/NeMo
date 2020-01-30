如何构建神经模块
================

.. note::
    目前，NeMo 只支持 Pytorch 作为后端

神经模块根据概念可以分成4个有重叠的类目:

* **Trainable Modules** - 包含了有可训练权重的模块。继承自
  :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` 类。
* **Data Layers** - 对数据做(extraction抽取, transform转换, load加载, feed接入) ETLF 操作的模块。继承自
  :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` 类。
* **Loss Modules** - 计算损失函数的模块。继承自
  :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` 类。
* **Non Trainable Modules** - 不可训练模块，比如，查表，数据增强，贪心解码器等。继承自
  :class:`NonTrainableNM<nemo.backends.pytorch.nm.NonTrainableNM>` 类。

在下面的图片中，你可以看到这些类的继承关系

.. figure:: nm_class_structure.png
   :alt: map to buried treasure

   继承类关系图。假设 API 的类是绿色的。红色类是用户将要执行的。

可训练模块
-----------------
.. note::
    注意 :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` 类
    有两个基础类：:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` 类 和 ``torch.nn.Module``.

从头定义模块
~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) 首先继承 :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` 类。
(2) 实现 ``input_ports`` 和 ``output_ports`` 属性，定义输入输出端口。

.. code-block:: python

    @property
    def input_ports(self):
        return {...}

    @property
    def output_ports(self):
        return {...}

(3) 在构造函数里，首先调用基类的构造函数

.. code-block:: python

    def __init__(self, module_params, ...)
        super().__init__()

(4) 实现 ``torch.nn.Module`` 模块里的 ``forward`` 方法 

.. important::
    你的 ``forward`` 方法的输入参数名必须匹配你的模块的输入的输入端口名。

例子 1
~~~~~~

.. code-block:: python

    class TaylorNet(TrainableNM): # (1) 注意继承 TrainableNM 类
        """学习Taylor系数的模块"""

        # (2) 定义输入输出端口
        @property
        def input_ports(self):
            return {"x": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)})}
        @property
        def output_ports(self):
            return {"y_pred": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)})}

        def __init__(self):
            # (3) 调用基类构造函数
            TrainableNM.__init__(self)
            # Neural Modules 的特定部分，剩下的是 PyTorch 代码
            self._dim = self.local_parameters["dim"]
            self.fc1 = nn.Linear(self._dim, 1)
            t.nn.init.xavier_uniform_(self.fc1.weight)
            self._device = t.device(
                "cuda" if self.placement == DeviceType.GPU else "cpu")
            self.to(self._device)

        # IMPORTANT: 给前向参数的名字必须匹配输入端口的名字
        def forward(self, x):
            # (4) Implement the forward method
            lst = []
            for pw in range(self._dim):
                lst.append(x**pw)
            nx = t.cat(lst, dim=-1)
            return self.fc1(nx)


转换 PyTorch 的 nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) 如果你已经有 PyTorch 的类继承自 ``torch.nn.Module`` ，把那个继承改成继承
    :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` 类。
(2) 实现 ``input_ports`` 和 ``output_ports`` 属性
(3) 修改构造函数，首先调用基类构造函数

.. code-block:: python

    class MyNeuralModule(TrainableNM):
        @property
        def input_ports(self):
            return {...}
        @property
        def output_ports(self):
            return {...}

        def __init__(self, module_params, ...)
            TrainableNM.__init__(self)

(4) 修改 ``forward`` 方法，使得它的输入参数和你的输入端口名字匹配。

数据层模块
----------
(1) 继承自 :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` 类。
(2) 实现 ``__len__`` 方法，返回数据集大小
(3) 实现 ``dataset`` 或者 ``data_iterator`` 属性，返回一个PyTorch数据集对象或者你的数据集的迭代器。(没有使用的属性应该返回None)

当实现构造函数的时候，你首先要调用基类构造函数，并且定义在output_ports定义 *仅输出端口* 。
另外，模块应该接收像是 ``batch_size`` 和 ``shuffle`` 的参数。

如果你使用了 ``torch.utils.data.Dataset`` 类 (*推荐方法*)，那么你可以实现 ``dataset`` 属性，一个数据加载器就会自动给你创建。
(见下面的例子)。

例子
~~~~

这个例子把 PyTorch 的 *ImageFolder* 数据集封装成一个神经模块的数据层。

.. code-block:: python

    import torch
    import torchvision
    import torchvision.transforms as transforms, datasets

    """这个类把 Pytorch 的 ImageFolder 数据集的 API 封装成了神经模块"""

    class ImageFolderDataLayer(DataLayerNM):
        @property
        def output_ports(self):
            # 注意，我们会定义输出的高和宽
            # 因此需要一个size参数
            return {
                "image": NeuralType(
                    {
                        0: AxisType(BatchTag),
                        1: AxisType(ChannelTag),
                        2: AxisType(HeightTag, self._input_size),
                        3: AxisType(WidthTag, self._input_size),
                    }
                ),
                "label": NeuralType({0: AxisType(BatchTag)}),
            }

        def __init__(self, input_size, path):
            DataLayerNM.__init__(self)

            self._input_size = input_size"
            self._path = path

            self._transforms = transforms.Compose([
                transforms.RandomResizedCrop(self._input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            self._dataset = datasets.ImageFolder(self._path, self._transforms)

        def __len__(self):
            return len(self._dataset)

        @property
        def dataset(self):
            return self._dataset

        @property
        def data_iterator(self):
            return None


损失函数神经模块
----------------

(1) 继承自 :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` 类
(2) 实现 ``input_ports`` 和 ``output_ports`` 属性
(3) 在构造函数里调用基类构造函数
(4) 实现 :meth:`_loss_function<nemo.backends.pytorch.nm.LossNM._loss_function>` 方法。

Example
~~~~~~~

.. code-block:: python

    class CrossEntropyLoss(LossNM):
        @property
        def input_ports(self):
            return {"predictions": NeuralType({
                        0: AxisType(BatchTag),
                        1: AxisType(ChannelTag)}),
                    "labels": NeuralType({
                        0: AxisType(BatchTag)
                        })
                    }

        @property
        def output_ports(self):
            return {"loss": NeuralType(None)}

        def __init__(self):
            # 神经模块 API
            super().__init__()

            # 结束神经模块 API
            self._criterion = torch.nn.CrossEntropyLoss()

        # 你需要实现这个方法
        def _loss_function(self, **kwargs):
            return self._criterion(*(kwargs.values()))
