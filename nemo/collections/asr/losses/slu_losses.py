import torch
import torch.nn as nn

from nemo.collections.asr.parts.utils.slu_utils import get_seq_mask
from nemo.core.classes import Loss


def get_seq_masked_loss(
    loss_fn, predictions, targets, lengths=None, label_smoothing=0.0, reduction="mean",
):
    """
    inputs:
    - predictions: BxTxC
    - targets: B
    - lengths: B
    """

    mask = torch.ones_like(predictions)
    if lengths is not None:
        mask = get_seq_mask(predictions, lengths).float()

    predictions = predictions.transpose(1, 2)  # BxTxC -> BxCxT

    loss = mask * loss_fn(predictions, targets)
    batch_size = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / batch_size
    elif reduction == "batch":
        loss = loss.reshape(batch_size, -1).sum(1) / mask.reshape(batch_size, -1).sum(1)

    if label_smoothing == 0.0:
        return loss
    else:
        # Regularizing Neural Networks by Penalizing Confident Output Distributions.
        # https://arxiv.org/abs/1701.06548
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss


class SeqNLLLoss(Loss):
    def __init__(self, reduction='mean', label_smoothing=0.0, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.nll_loss = nn.NLLLoss(reduction='none', **kwargs)

    def forward(self, log_probs, targets, lengths):
        """
        inputs:
        - log_probs: BxTxC
        - targets: B
        - lengths: B
        """
        loss = get_seq_masked_loss(
            loss_fn=self.nll_loss,
            predictions=log_probs,
            targets=targets,
            lengths=lengths,
            label_smoothing=self.label_smoothing,
        )
        return loss
