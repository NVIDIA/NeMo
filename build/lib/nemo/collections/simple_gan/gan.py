# Copyright (c) 2019 NVIDIA Corporation
"""A collection of Neural Modules to be used for training a WGAN-GP on MNIST"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from nemo.backends.pytorch.nm import DataLayerNM, LossNM, TrainableNM
from nemo.core import DeviceType
from nemo.core.neural_types import ChannelType, LabelsType, LossType, NeuralType
from nemo.utils.decorators import add_port_docs


class SimpleDiscriminator(TrainableNM):
    """Simple convolutional discrimnator that takes in a 28x28 greyscale image
    and assigns a score to it.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "image": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # )
            "image": NeuralType(('B', 'C', 'H', 'W'), ChannelType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"decision": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag, 1)})}
        return {"decision": NeuralType(('B', 'C'), ChannelType())}

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.fc_layer = torch.nn.Linear(256 * 4 * 4, 1)
        self.to(self._device)

    def forward(self, image):
        decision = self.layers(image)
        decision = decision.view(-1, 256 * 4 * 4)
        decision = self.fc_layer(decision)
        return decision


class SimpleGenerator(TrainableNM):
    """Simple convolutional generator that takes a random variable of size
    (64, 4, 4) and produces a 28x28 greyscale image.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "latents": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag, 64),
            #         2: AxisType(HeightTag, 4),
            #         3: AxisType(WidthTag, 4),
            #     }
            # )
            "latents": NeuralType(('B', 'C', 'H', 'W'), ChannelType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "image": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # )
            "image": NeuralType(('B', 'C', 'H', 'W'), ChannelType())
        }

    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 128, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 1, 12),
            torch.nn.Sigmoid(),
        )
        self.to(self._device)

    def forward(self, latents):
        image = latents
        for layer in self.layers:
            image = layer(image)
        return image


class DiscriminatorLoss(LossNM):
    """Computes the loss from a disciminator score by simply taking the mean
    of all scores in a batch.

    Args:
        neg (bool): Whether to negate the final loss
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.

        decision:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag, 1)
        """
        return {
            # "decision": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag, 1)}),
            "decision": NeuralType(('B', 'D'), ChannelType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, neg=False):
        super().__init__()
        self.neg = neg

    def _loss(self, decision):
        if self.neg:
            return -torch.mean(decision)
        return torch.mean(decision)

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))


class GradientPenalty(LossNM):
    """Compute the gradient penalty of the disciminator

    Args:
        lambda_ (float): lambda parameter indicating the weight of the loss.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "interpolated_image": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # ),
            # "interpolated_decision": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag, 1)}),
            "interpolated_image": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "interpolated_decision": NeuralType(('B', 'C'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def _loss(self, interpolated_image, interpolated_decision):
        grad_outputs = torch.ones(interpolated_decision.size(), dtype=interpolated_image.dtype)
        if self.placement != DeviceType.CPU:
            grad_outputs = grad_outputs.cuda()

        gradients = torch.autograd.grad(
            outputs=interpolated_decision,
            inputs=interpolated_image,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_ * gradient_penalty

    def _loss_function(self, **kwargs):
        return self._loss(**kwargs)


class InterpolateImage(TrainableNM):
    """Linearly interpolates an image between image1 and image2
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "image1": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # ),
            # "image2": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # ),
            "image1": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "image2": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "interpolated_image": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, 28),
            #         3: AxisType(WidthTag, 28),
            #     }
            # )
            "interpolated_image": NeuralType(('B', 'C', 'H', 'W'), ChannelType())
        }

    def __init__(self):
        super().__init__()

    def forward(self, image1, image2):
        alpha = torch.rand(image1.shape[0], 1).unsqueeze(-1).unsqueeze(-1)
        alpha = alpha.to(self._device)
        alpha = alpha.type(image2.dtype)
        image1 = image1.type(image2.dtype)

        interpolated_image = alpha * image1 + ((1 - alpha) * image2)
        return torch.autograd.Variable(interpolated_image, requires_grad=True)


class RandomDataLayer(DataLayerNM):
    """Dummy data layer for return random variables to be used in the generator

    Args:
        batch_size (int)
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        latent:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag, 64)

            2: AxisType(HeightTag, 4)

            3: AxisType(WidthTag, 4)
        """
        return {
            # "latent": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag, 64),
            #         2: AxisType(HeightTag, 4),
            #         3: AxisType(WidthTag, 4),
            #     }
            # )
            "latent": NeuralType(('B', 'C', 'H', 'W'), ChannelType())
        }

    def __init__(self, batch_size):
        super().__init__()
        self._batch_size = batch_size

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, batch_size):
                super().__init__()
                self._batch_size = batch_size

            def __getitem__(self, i):
                return torch.randn(64, 4, 4)

            def __len__(self):
                return self._batch_size * 2

        self._dataset = DummyDataset(batch_size)

    def __len__(self):
        return self._dataset.__len__()

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class MnistGanDataLayer(DataLayerNM):
    """Wrapper around torchvision's MNIST dataset. Additionally, it returns a
    random variable to be used in the generator.

    Args:
        batch_size (int)
        root (str): Where to store the dataset
        train (bool)
        shuffle (bool)
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "latent": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag, 64),
            #         2: AxisType(HeightTag, 4),
            #         3: AxisType(WidthTag, 4),
            #     }
            # ),
            # "image": NeuralType(
            #     {
            #         0: AxisType(BatchTag),
            #         1: AxisType(ChannelTag),
            #         2: AxisType(HeightTag, self._input_size[1]),
            #         3: AxisType(WidthTag, self._input_size[0]),
            #     }
            # ),
            # "label": NeuralType({0: AxisType(BatchTag)}),
            "latent": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "image": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "label": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(self, batch_size, root, train=True, shuffle=True):
        super().__init__()
        self._input_size = (28, 28)

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle
        self._root = root
        self._transforms = transforms.Compose([transforms.ToTensor()])

        self._dataset = datasets.MNIST(root=self._root, train=self._train, download=True, transform=self._transforms)

        class DatasetWrapper(Dataset):
            def __init__(self, dataset):
                super().__init__()
                self._dataset = dataset

            def __getitem__(self, index):
                latents = torch.randn(64, 4, 4)
                items = self._dataset.__getitem__(index)
                return latents, items[0], items[1]

            def __len__(self):
                return self._dataset.__len__()

        self._dataset = DatasetWrapper(self._dataset)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
