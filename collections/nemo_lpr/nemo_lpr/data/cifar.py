# Copyright (c) 2019 NVIDIA Corporation
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core import *
from torchvision import transforms, datasets


class CIFAR10DataLayer(DataLayerNM):
    """This class wraps Torchvision's CIFAR data set API into NeuralModule."""

    @staticmethod
    def create_ports(input_size=(32, 32)):
        input_ports = {}
        output_ports = {
            "image": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(ChannelTag),
                                 2: AxisType(HeightTag, input_size[1]),
                                 3: AxisType(WidthTag, input_size[0])}),
            "label": NeuralType({0: AxisType(BatchTag)}),
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
        self._input_size = (32, 32)
        create_port_args = {"input_size": self._input_size}
        DataLayerNM.__init__(self, create_port_args=create_port_args, **kwargs)

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle
        self._root = root
        self._transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._dataset = datasets.CIFAR10(root=self._root, train=self._train,
                                         download=True,
                                         transform=self._transforms)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        """
        if self.placement == DeviceType.AllGpu:
          self.train_sampler =
          t.utils.data.distributed.DistributedSampler(self._dataset)
        else:
          self.train_sampler = None

        return t.utils.data.DataLoader(self._dataset,
            batch_size=self._batch_size,
            shuffle=(self.train_sampler == None), num_workers=4,
            sampler=self.train_sampler)
        """
        return None
