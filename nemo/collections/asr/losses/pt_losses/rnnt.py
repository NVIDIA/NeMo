import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import (
    LossType,
    NeuralType,
    SpectrogramType,
    VoidType,
    LabelsType,
    LengthsType,
    LogprobsType,
)
from nemo.collections.asr.losses.rnnt import RNNTLoss


__all__ = ["RNNTLossForSSL"]


class RNNTLossForSSL(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return True

    def __init__(self, num_classes):
        super().__init__()
        self.loss = RNNTLoss(num_classes=num_classes)

    @typecheck()
    def forward(self, spec_masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):

        loss = self.loss(
            log_probs=decoder_outputs, targets=targets, input_lengths=decoder_lengths, target_lengths=target_lengths
        )

        return loss
