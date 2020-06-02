# Copyright (c) 2019 NVIDIA Corporation
from torchvision import datasets, transforms

from .....core import *
from ...nm import DataLayerNM
from nemo.utils.decorators import add_port_docs


class ImageFolderDataLayer(DataLayerNM):
    """This class wraps Torchvision's ImageFolder data set API into
    NeuralModule."""

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
            #         2: AxisType(HeightTag, self._input_size),
            #         3: AxisType(WidthTag, self._input_size),
            #     }
            # ),
            # "label": NeuralType({0: AxisType(BatchTag)}),
            "image": NeuralType(elements_type=ChannelType(), axes=('B', 'C', 'H', 'W')),
            "label": NeuralType(elements_type=LogitsType(), axes=tuple('B')),
        }

    def __init__(self, batch_size, path, input_size=32, shuffle=True, is_eval=False):
        super().__init__()

        self._input_size = input_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._path = path
        self._eval = is_eval

        if not self._eval:
            self._transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self._input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        else:
            # These are tuned for ImageNet evaluation
            self._transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(self._input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        self._dataset = datasets.ImageFolder(self._path, self._transforms)

    def __len__(self):
        return len(self._dataset)

    # @property
    # def data_iterator(self):
    #   if self.placement == DeviceType.AllGpu:
    #     self.train_sampler = t.utils.data.distributed.DistributedSampler(
    #     self._dataset)
    #   else:
    #     self.train_sampler = None
    #   return t.utils.data.DataLoader(self._dataset,
    #   batch_size=self._batch_size,
    #                                  shuffle=(self.train_sampler == None),
    #                                  num_workers=4,
    #                                  sampler=self.train_sampler)
    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
