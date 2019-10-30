import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


import nemo

from nemo.backends.pytorch.nm import TrainableNM, NonTrainableNM, LossNM,\
    DataLayerNM
from nemo.core import NeuralType, BatchTag, ChannelTag, HeightTag, WidthTag,\
    AxisType, DeviceType


class MNISTDataLayer(DataLayerNM):
    """Wrapper around torchvision's MNIST dataset.

    Args:
        batch_size (int)
        root (str): Where to store the dataset
        train (bool)
        shuffle (bool)
    """

    @staticmethod
    def create_ports(input_size=(32, 32)):
        input_ports = {}
        output_ports = {
            "image": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(ChannelTag, 1),
                                 2: AxisType(HeightTag, input_size[1]),
                                 3: AxisType(WidthTag, input_size[0])}),
            "label": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            batch_size,
            root,
            train=True,
            shuffle=True,
            **kwargs
    ):
        # Passing the default params to base init call.
        DataLayerNM.__init__(self, **kwargs)

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle
        self._root = root

        # Up-scale and transform to tensors.
        self._transforms = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])

        self._dataset = datasets.MNIST(root=self._root, train=self._train,
                                       download=True,
                                       transform=self._transforms)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class LeNet5(TrainableNM):
    """Classical LeNet-5 model for MNIST image classification.
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 1),
                                  2: AxisType(HeightTag, 32),
                                  3: AxisType(WidthTag, 32)
                                  })
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(ChannelTag)
                                       })
            # no PredictionTag!?

        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the LeNet-5 model.
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            # reshape to [-1, 120]
            torch.nn.Flatten(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
            torch.nn.LogSoftmax(dim=1)
        )
        self.to(self._device)

    def forward(self, images):
        predictions = self.model(images)
        return predictions


# 0. Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory()

# 1. Instantiate necessary neural modules
dl = MNISTDataLayer(
    batch_size=64,
    root="~/data/mnist",
    train=True,
    shuffle=True
)

lenet5 = LeNet5()

ce_loss = nemo.tutorials.CrossEntropyLoss()

# 2. Describe activation's flow
x, y = dl()
p = lenet5(images=x)
loss = ce_loss(predictions=p, labels=y)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss],
    print_func=lambda x: print(f'Train Loss: {str(x[0].item())}'))

# Invoke "train" action
nf.train([loss], callbacks=[callback],
         optimization_params={"num_epochs": 10, "lr": 0.0001},
         optimizer="adam")

# How
