# Copyright (c) 2019 NVIDIA Corporation
import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


class CTCLossNM(LossNM):
    """
    Neural Module wrapper for pytorch's ctcloss

    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        log_probs:
            1: AxisType(TimeTag)

            0: AxisType(BatchTag)

            2: AxisType(ChannelTag)

        targets:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_length:
            0: AxisType(BatchTag)

        target_length:
            0: AxisType(BatchTag)
        """
        return {
            "log_probs": NeuralType({1: AxisType(TimeTag), 0: AxisType(BatchTag), 2: AxisType(ChannelTag),}),
            "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_length": NeuralType({0: AxisType(BatchTag)}),
            "target_length": NeuralType({0: AxisType(BatchTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, num_classes):
        super().__init__()

        # self._blank = self.local_parameters.get('blank', 0)
        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none')

    def _loss(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length, target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))
