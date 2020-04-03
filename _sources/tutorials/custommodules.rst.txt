How to build Neural Module
==========================

.. note::
    Currently, NeMo only supports PyTorch as a backend.

Neural Modules can be conceptually classified into 4 potentially overlapping categories:

* **Trainable Modules** - modules that contain trainable weights. Inherit from
  :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
* **Data Layers** - modules that perform (extract, transform, load, feed) ETLF of the data. Inherit from
  :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` class.
* **Loss Modules** - modules that compute loss functions. Inherit from
  :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` class.
* **Non Trainable Modules** - non-trainable module, for example, table lookup, data augmentation, greedy decoder, etc. Inherit from
  :class:`NonTrainableNM<nemo.backends.pytorch.nm.NonTrainableNM>` class.

In the figure below you can see a class inheritance diagram for these helper classes.

.. figure:: nm_class_structure.png
   :alt: map to buried treasure

   Inheritance class diagram. Provided API's classes are in green. Red classes are to be implemented by user.

Trainable Module 
-----------------
.. note::
    Notice that :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class
    has two base classes: :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` class and ``torch.nn.Module``.

Defining a module from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) Inherit from :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
(2) Create the ``input_ports`` and ``output_ports`` properties that define your input and output ports.

.. code-block:: python

    @property
    def input_ports(self):
        return {...}

    @property
    def output_ports(self):
        return {...}

(3) In the constructor, call base class constructor first

.. code-block:: python

    def __init__(self, module_params, ...)
        super().__init__()

(4) Implement ``forward`` method from ``torch.nn.Module``

.. important::
    Input argument names to your ``forward`` method must match your module's input port names exactly.

Example 1
~~~~~~~~~

.. code-block:: python

    class TaylorNet(TrainableNM): # (1) Note inheritance from TrainableNM
        """Module which learns Taylor's coefficients."""

        # (2) Code to define input and output ports
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

        def __init__(self, dim):
            # (3) Call base constructor
            super().__init__()
            # And of Neural Modules specific part. Rest is PyTorch code
            self._dim = dim
            self.fc1 = nn.Linear(self._dim, 1)
            t.nn.init.xavier_uniform_(self.fc1.weight)
            self._device = t.device(
                "cuda" if self.placement == DeviceType.GPU else "cpu")
            self.to(self._device)

        # IMPORTANT: input arguments to forward must match input ports' names
        def forward(self, x):
            # (4) Implement the forward method
            lst = []
            for pw in range(self._dim):
                lst.append(x**pw)
            nx = t.cat(lst, dim=-1)
            return self.fc1(nx)



Converting from PyTorch's nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) If you already have a PyTorch class which inherits from ``torch.nn.Module``, replace that inheritance with inheritance from
    :class:`TrainableNM<nemo.backends.pytorch.nm.TrainableNM>` class.
(2) Implement the ``input_ports`` and ``output_ports`` properties
(3) Modify your constructor to call the base class constructor first.

.. code-block:: python

    class MyNeuralModule(TrainableNM):
        @property
        def input_ports(self):
            return {...}
        @property
        def output_ports(self):
            return {...}

        def __init__(self, module_params, ...)
            super().__init__()

(4) Modify ``forward`` method so that its input arguments match your input port names exactly.

Data Layer Module
------------------------
(1) Inherit from :class:`DataLayerNM<nemo.backends.pytorch.nm.DataLayerNM>` class.
(2) Implement ``__len__`` method to return dataset size.
(3) Implement either the ``dataset`` or ``data_iterator`` property to return a PyTorch Dataset object or an iterator over your dataset, respectively. (The unused property should return None.)

When implementing the constructor, you should first call the base class constructor and
define *output ports only* in ``output_ports``.  Also, module should accept
parameters such as ``batch_size`` and ``shuffle``.

If you are using ``torch.utils.data.Dataset`` class (*recommended approach*), then you can implement the ``dataset`` property, and a DataLoader will be created for you.
Here is an example:

Example
~~~~~~~

This example wraps PyTorch's *ImageFolder* dataset into a neural module data layer.


.. code-block:: python
  
    import torch 
    import torchvision
    import torchvision.transforms as transforms, datasets

    """This class wraps Torchvision's ImageFolder data set API into NeuralModule."""

    class ImageFolderDataLayer(DataLayerNM):

        @property
        def output_ports(self):
            """Returns definitions of module output ports."""
            # Note: we define the size of the height and width of our output
            # tensors, and thus require a size parameter.
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
            super().__init__()

            self._input_size = input_size
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


Loss Neural Module
------------------

(1) Inherit from :class:`LossNM<nemo.backends.pytorch.nm.LossNM>` class
(2) Create ports using the ``input_ports`` and ``output_ports`` properties
(3) In your constructor, call base class constructor
(4) Implement :meth:`_loss_function<nemo.backends.pytorch.nm.LossNM._loss_function>` method.


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
            # Neural Module API specific
            super().__init__()

            # End of Neural Module API specific
            self._criterion = torch.nn.CrossEntropyLoss()

        # You need to implement this function
        def _loss_function(self, **kwargs):
            return self._criterion(*(kwargs.values()))
