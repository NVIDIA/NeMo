import torch

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


class TRADEMaskedCrosEntropy(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "logits": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag), 3: AxisType(ChannelTag)}
            ),
            "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag), 2: AxisType(TimeTag)}),
            "mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self):
        LossNM.__init__(self)

    def _loss_function(self, logits, targets, mask):
        logits_flat = logits.view(-1, logits.size(-1))
        eps = 1e-10
        log_probs_flat = torch.log(torch.clamp(logits_flat, min=eps))
        target_flat = targets.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*targets.size())
        loss = self.masking(losses, mask)
        return loss

    @staticmethod
    def masking(losses, mask):

        # mask_ = []
        # batch_size = mask.size(0)
        max_len = losses.size(2)

        mask_ = torch.arange(max_len, device=mask.device)[None, None, :] < mask[:, :, None]
        # mask_ = torch.arange(max_len, device=mask.device).expand(losses.size()) < mask.expand(losses)

        # for si in range(mask.size(1)):
        #     seq_range = torch.arange(0, max_len).long()
        #     seq_range_expand = \
        #         seq_range.unsqueeze(0).expand(batch_size, max_len)
        #     if mask[:, si].is_cuda:
        #         seq_range_expand = seq_range_expand.cuda()
        #     seq_length_expand = mask[:, si].unsqueeze(
        #         1).expand_as(seq_range_expand)
        #     mask_.append((seq_range_expand < seq_length_expand))
        # mask_ = torch.stack(mask_)
        # mask_ = mask_.transpose(0, 1)
        # if losses.is_cuda:
        #     mask_ = mask_.cuda()

        mask_ = mask_.float()
        losses = losses * mask_
        loss = losses.sum() / mask_.sum()
        return loss
