import torch
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.utils.decorators import experimental
from nemo.core.classes import Loss

#TODO
@experimental
class GlowTTSLoss(Loss):
    """
    GlowTTSLoss
    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        zero_infinity (bool): Whether to zero infinite losses and the associated gradients.
            By default, it is False. Infinite losses mainly occur when the inputs are too
            short to be aligned to the targets.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False):
        super().__init__()

        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none', zero_infinity=zero_infinity)

    @typecheck()
    def forward(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length, target_length)
        loss = torch.mean(loss)
        return loss