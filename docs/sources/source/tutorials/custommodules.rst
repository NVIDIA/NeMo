How to build Neural Module
==========================

.. note::
    Currently, NeMo only support PyTorch as a backend.

Neural Modules can be conceptually classified into 4 potentially overlapping categories:

* **Trainable Modules** - modules that contain trainable weights. Inherit from
  :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
* **Data Layers** - modules that perform (extract, transform, load, feed) ETLF of the data. Inherit from
  :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` class.
* **Loss Modules** - modules that compute loss functions. Inherit from
  :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` class.
* **Non Trainable Modules** - non-trainable module, for example, table lookup, data augmentation, greedy decoder, etc. Inherit from
  :class:`NonTrainableNM<nemo.backends.pytorch.nm.NonTrainableNM>` class.

In Figure below you can see a class inheritance diagram for these helper classes.

.. figure:: nm_class_structure.png
   :alt: map to buried treasure

   Inheritance class diagram. Provided API's classes are in green. Red classes are to be implemented by user.

Trainable Module 
-----------------
.. note::
    Notice that :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class
    has two base classes: :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` class and ``torch.nn.Module``.

Define module from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) Inherit from :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
(2) Create the ``create_ports()`` static method that defines your input and output ports.
    If your ``create_ports()`` method requires some params, pass it to the base class
    constructor as part of the ``create_port_args`` param.

.. code-block:: python

    @staticmethod
    def create_ports(size):
        input_ports = {...}
        output_ports = {...}
        return input_ports, output_ports

(3) In the constructor, call base class constructor first

.. code-block:: python

    def __init__(self, *, module_params, .., size, **kwargs)
        super().__init__(create_port_args={"size": size}, **kwargs)

(4) Implement ``forward`` method from ``torch.nn.Module``

.. important::
    Input argument names to your ``forward`` method must match your module's input port names exactly.

Example 1
~~~~~~~~~

.. code-block:: python

    class TaylorNet(TrainableNM): # (1) Note inheritance from TrainableNM
        """Module which learns Taylor's coefficients."""

        # (2) Code create_ports() to define input and output ports
        @staticmethod
        def create_ports():
            input_ports = {"x": NeuralType({0: AxisType(BatchTag),
                                            1: AxisType(ChannelTag)})}
            output_ports = {"y_pred": NeuralType({0: AxisType(BatchTag),
                                                  1: AxisType(ChannelTag)})}
            return input_ports, output_ports

        def __init__(self, **kwargs):
            # (3) call base constructor
            TrainableNM.__init__(self, **kwargs)
            # And of Neural Modules specific part. Rest is PyTorch code
            self._dim = self.local_parameters["dim"]
            self.fc1 = nn.Linear(self._dim, 1)
            t.nn.init.xavier_uniform_(self.fc1.weight)

        # IMPORTANT: input arguments to forward must match input input ports' names
        def forward(self, x):
            lst = []
            for pw in range(self._dim):
                lst.append(x**pw)
            nx = t.cat(lst, dim=-1)
            return self.fc1(nx)



Converting from PyTorch's nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) If you already have PyTorch class which inherits from ``torch.nn.Module``, replace that inheritance with inheritance from
    :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
(2) Create the ``create_ports()`` static method
(3) Modify your constructor to call base class constructor first.

.. code-block:: python

    class MyNeuralModule(TrainableNM):
        @staticmethod
        def create_ports():
            input_ports = {...}
            output_ports = {...}
            return input_ports, output_ports

        def __init__(self, *, module_params, .., **kwargs)
            TrainableNM.__init__(self, **kwargs)

(4) Modify ``forward`` method so that its input arguments match your input port names exactly.

Example 2: TODO
~~~~~~~~~~~~~~~



Data Layer Module
------------------------
(1) Inherit from :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` class.
(2) Implement ``__len__`` method to return dataset size
(3) Implement ``data_iterator`` property to return iterator over your dataset.

When implementing constructor, you should first call base class constructor and
define *output ports only* in create_ports().  Also, module should accept
parameters such as ``batch_size`` and ``shuffle``.

If under the hood you are using ``torch.utils.data.Dataset`` class (*recommended approach*), then you can use ``torch.utils.data.DataLoader`` to construct iterator over your data.
(See example below).

Example
~~~~~~~

This example wraps PyTorch's *ImageFolder* dataset into a neural module data layer.


.. code-block:: python
  
    import torch 
    import torchvision
    import torchvision.transforms as transforms, datasets

    """This class wraps Torchvision's ImageFolder data set API into NeuralModule."""

    class ImageFolderDataLayer(DataLayerNM):
        @staticmethod
        def create_ports(size):
            # Note: we define the size of the height and width of our output
            # tensors, and thus require a size parameter.
            input_ports = {}
            self._output_ports = {
                "image": NeuralType({0: AxisType(BatchTag),
                                     1: AxisType(ChannelTag),
                                     2: AxisType(HeightTag, size),
                                     3: AxisType(WidthTag, size)}),
                "label": NeuralType({0: AxisType(BatchTag)})
            }
            return input_ports, output_ports

        def __init__(self, **kwargs):
            create_port_args = {"size": kwargs["input_size"]}
            DataLayerNM.__init__(self, create_port_args=create_port_args, **kwargs)

            self._batch_size = kwargs["batch_size"]
            self._input_size = kwargs["input_size"]
            self._shuffle = kwargs.get("shuffle", True)
            self._path = kwargs["path"]

            self._device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")

            self._transforms = transforms.Compose([
                transforms.RandomResizedCrop(self._input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            self._dataset = datasets.ImageFolder(self._path, self._transforms)

        def __len__(self):
            return len(self._dataset)

        @property
        def data_iterator(self):
            return torch.utils.data.DataLoader(self._dataset,
                                           batch_size=self._batch_size,
                                           shuffle=self._shuffle)


Loss Neural Module
------------------

(1) Inherit from :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` class
(2) Create create_ports() method
(3) In your constructor, call base class constructor
(4) Implement :meth:`_loss_function<nemo.backends.pytorch.nm.LossNM._loss_function>` method.


Example
~~~~~~~

.. code-block:: python

    class CrossEntropyLoss(LossNM):
        @staticmethod
        def create_ports():
            input_ports = {"predictions": NeuralType({0: AxisType(BatchTag),
                                                      1: AxisType(ChannelTag)}),
                           "labels": NeuralType({0: AxisType(BatchTag)})}
            output_ports = {"loss": NeuralType(None)}
            return input_ports, output_ports

        def __init__(self, **kwargs):
            # Neural Module API specific
            super().__init__(**kwargs)

            # End of Neural Module API specific
            self._criterion = torch.nn.CrossEntropyLoss()

        # You need to implement this function
        def _loss_function(self, **kwargs):
            return self._criterion(*(kwargs.values()))


