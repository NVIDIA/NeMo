import torch
from torch import nn

from nemo.backends.pytorch.nm import LossNM
from nemo.core import AxisType, BatchTag, ChannelTag, NeuralType, RegressionTag, TimeTag

__all__ = ['SequenceLoss', 'CrossEntropyLoss', 'MSELoss']

EPS = 1e-5


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

        log_probs:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        targets:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            'log_probs': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),}),
            'targets': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)

        """
        return {"loss": NeuralType(None)}

    def __init__(
        self,
        pad_id=0,
        smoothing_coef=0.0,
        sample_wise=False,
        aux_ctc=False,
        ctc_initial_coef=0.1,
        ctc_blank_id=None,
        **kwargs
    ):
        assert (not aux_ctc) or (ctc_blank_id is not None), "Should be a blank id if using CTC loss"

        super().__init__(**kwargs)

        self.pad_id = pad_id
        self.smoothing_coef = smoothing_coef
        self.sample_wise = sample_wise
        self.aux_ctc = aux_ctc
        self.ctc_coef = ctc_initial_coef

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
            loss /= pad_mask.sum() + EPS
        return loss

    def _ctc_loss(self, log_probs, targets, pad_mask):
        lengths = pad_mask.sum(-1)
        loss = self.ctc(log_probs.transpose(0, 1), targets, lengths, lengths)
        loss = torch.mean(loss)
        return loss


class CrossEntropyLoss(LossNM):
    """
    CrossEntropyLoss

    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        labels:
            0: AxisType(BatchTag)

        """
        return {
            "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "labels": NeuralType({0: AxisType(BatchTag),}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, weight=None, **kwargs):
        LossNM.__init__(self, **kwargs)
        if weight:
            weight = torch.FloatTensor(weight).to(self._device)
        self._criterion = nn.CrossEntropyLoss(weight=weight)

    def _loss_function(self, logits, labels):
        loss = self._criterion(logits, labels)
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
            "preds": NeuralType({0: AxisType(RegressionTag)}),
            "labels": NeuralType({0: AxisType(RegressionTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.MSELoss()

    def _loss_function(self, preds, labels):
        loss = self._criterion(preds, labels)
        return loss
