import torch
from torch import nn

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import LabelsType, LogitsType, LossType, MaskType, NeuralType, RegressionValuesType

__all__ = ['SequenceLoss', 'CrossEntropyLoss', 'MSELoss']


class SequenceLoss(LossNM):
    """Loss for seq2seq tasks

    Args:
        pad_id (int): Label position of padding symbol.
            Defaults to 0.
        smoothing_coef (float): Label smoothing coefficient in range [0, 1].
            Defaults to 0.0.
        sample_wise (bool): Flag indicates if loss sum divisor should be batch
            size.
            Defaults to False.
        aux_ctc (bool): Whether to add auxiliary CTC loss.
            Defaults to False.
        ctc_initial_coef (float): Initial coefficient to multiply ctc component
            by.
            Defaults to 0.1.
        ctc_blank_id (int): ID of blank symbols to pass to mask when
            calculating ctc loss.
            Defaults to None.

    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {'log_probs': NeuralType(axes=('B', 'T', 'D')), 'targets': NeuralType(axes=('B', 'T'))}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)

        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        pad_id=0,
        smoothing_coef=0.0,
        sample_wise=False,
        aux_ctc=False,
        ctc_initial_coef=0.1,
        ctc_blank_id=None,
        eps=1e-5,
    ):
        assert (not aux_ctc) or (ctc_blank_id is not None), "Should be a blank id if using CTC loss"

        super().__init__()

        self.pad_id = pad_id
        self.smoothing_coef = smoothing_coef
        self.sample_wise = sample_wise
        self.aux_ctc = aux_ctc
        self.ctc_coef = ctc_initial_coef
        self.eps = eps

        if aux_ctc:
            self.ctc = nn.CTCLoss(blank=ctc_blank_id, reduction='none', zero_infinity=True)
            self.ctc = self.ctc.to(self._device)

    def _loss_function(self, log_probs, targets):
        """(BTC, BT) -> 0"""

        pad_mask = (targets != self.pad_id).long()
        loss = self._ce_loss(log_probs, targets, pad_mask)

        if self.aux_ctc:
            ctc_loss = self._ctc_loss(log_probs, targets, pad_mask)
            loss += self.ctc_coef * ctc_loss

        assert loss.dim() == 0, "Zero-dim tensor check"

        return loss

    def _ce_loss(self, log_probs, targets, pad_mask):
        target_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
        loss = (1.0 - self.smoothing_coef) * target_log_probs + self.smoothing_coef * log_probs.mean(-1)
        pad_mask = pad_mask.float()
        loss = -torch.sum(loss * pad_mask)
        if self.sample_wise:
            loss /= target_log_probs.size(0)
        else:
            loss /= pad_mask.sum() + self.eps
        return loss

    def _ctc_loss(self, log_probs, targets, pad_mask):
        lengths = pad_mask.sum(-1)
        loss = self.ctc(log_probs.transpose(0, 1), targets, lengths, lengths)
        loss = torch.mean(loss)
        return loss


class CrossEntropyLoss(LossNM):
    """
    CrossEntropyLoss
    Args:
        logits_dim (int): dimension size of the logits tensor
        logits (float): output of the classifier
        labels (long): ground truth labels
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 1), LogitsType()),
            "labels": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), LabelsType()),
            "loss_mask": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), MaskType(), optional=True),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, weight=None, reduce=True):
        super().__init__()

        if weight:
            weight = torch.FloatTensor(weight).to(self._device)
        self._criterion = nn.CrossEntropyLoss(weight=weight, reduce=reduce)

    def _loss_function(self, logits, labels, loss_mask=None):
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-2)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        loss = self._criterion(logits_flatten, labels_flatten)
        return loss


class MSELoss(LossNM):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        preds:
            0: AxisType(RegressionTag)

        labels:
            0: AxisType(RegressionTag)
        """
        return {
            "preds": NeuralType(tuple('B'), RegressionValuesType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self):
        super().__init__()
        self._criterion = nn.MSELoss()

    def _loss_function(self, preds, labels):
        loss = self._criterion(preds, labels)
        return loss


#
# class XEntropyLoss(LossNM):
#     @property
#     def input_ports(self):
#         """Returns definitions of module input ports.
#         """
#         return {
#             "logits": NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
#             "labels": NeuralType(('B', 'D', 'T'), LabelsType()),
#             "length_mask": NeuralType(('B', 'D'), LengthsType()),
#         }
#
#     @property
#     def output_ports(self):
#         """Returns definitions of module output ports.
#
#         loss:
#             NeuralType(None)
#         """
#         return {"loss": NeuralType(axes=None, elements_type=LossType())}
#
#     def __init__(self):
#         super().__init__()
#
#     def _loss_function(self, inp, target, mask):
#         inp = inp.view(-1, inp.shape[2])
#         mask = mask.view(-1, 1)
#         crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
#         loss = crossEntropy.masked_select(mask).mean()
#         loss = loss.to(self._device)
#         return loss
