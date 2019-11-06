import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


import nemo
import torch

from nemo.backends.pytorch.nm import TrainableNM, NonTrainableNM, LossNM,\
    DataLayerNM
from nemo.core import NeuralType, BatchTag, ChannelTag, HeightTag, WidthTag,\
    AxisType, DeviceType, LogProbabilityTag


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
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 1),
                                  2: AxisType(HeightTag, input_size[1]),
                                  3: AxisType(WidthTag, input_size[0])}),
            "targets": NeuralType({0: AxisType(BatchTag)})
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
                                       1: AxisType(LogProbabilityTag)
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


class NLLLoss(LossNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "predictions": NeuralType(
                {0: AxisType(BatchTag),
                 1: AxisType(LogProbabilityTag)}),
            "targets": NeuralType({0: AxisType(BatchTag)}),
        }
        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        # Neural Module API specific
        LossNM.__init__(self, **kwargs)
        # End of Neural Module API specific
        self._criterion = torch.nn.NLLLoss()

    # You need to implement this function
    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


# 0. Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory(placement=DeviceType.CPU)

# 1. Instantiate necessary neural modules
dl = MNISTDataLayer(
    batch_size=64,
    root="~/data/mnist",
    train=True,
    shuffle=True
)

lenet5 = LeNet5()

nll_loss = NLLLoss()

# 2. Describe activation's flow
x, y = dl()
p = lenet5(images=x)
loss = nll_loss(predictions=p, targets=y)

####
dl_e = MNISTDataLayer(
    batch_size=64,
    root="~/data/mnist",
    train=False,
    shuffle=True
)
x, y = dl_e()
p = lenet5(images=x)
nll_loss_e = NLLLoss()
loss_e = nll_loss_e(predictions=p, targets=y)


def eval_iter_callback(tensors, global_vars):
    if "eval_loss" not in global_vars.keys():
        global_vars["eval_loss"] = []
    for kv, v in tensors.items():
        if kv.startswith("loss"):
            global_vars["eval_loss"].append(torch.mean(torch.stack(v)))
            # global_vars['eval_loss'].append(v.item())


def eval_epochs_done_callback(global_vars):
    eloss = torch.max(torch.tensor(global_vars["eval_loss"]))
    print("Evaluation Loss: {0}".format(eloss))
    return dict({"Evaluation Loss": eloss})


ecallback = nemo.core.EvaluatorCallback(eval_tensors=[loss_e],
                                        user_iter_callback=eval_iter_callback,
                                        user_epochs_done_callback=eval_epochs_done_callback,
                                        eval_step=100)

###########


# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss],
    print_func=lambda x: print(f'Train Loss: {str(x[0].item())}'))


# Invoke "train" action
nf.train([loss], callbacks=[callback, ecallback],
         optimization_params={"num_epochs": 10, "lr": 0.001},
         optimizer="adam")

# How
