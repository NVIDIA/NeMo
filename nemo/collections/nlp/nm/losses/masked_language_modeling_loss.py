from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.nm.losses.smoothed_cross_entropy_loss import SmoothedCrossEntropyLoss
from nemo.core import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


class MaskedLanguageModelingLossNM(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        output_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            "output_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, label_smoothing=0.0, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = SmoothedCrossEntropyLoss(label_smoothing)

    def _loss_function(self, logits, output_ids, output_mask):
        loss = self._criterion(logits, output_ids, output_mask)
        return loss
