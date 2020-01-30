from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.nm.losses.smoothed_cross_entropy_loss import SmoothedCrossEntropyLoss
from nemo.collections.nlp.utils.nlp_utils import mask_padded_tokens
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag, ChannelTag


class PaddedSmoothedCrossEntropyLossNM(LossNM):
    """
    Neural module which calculates CrossEntropyLoss and
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens

    Args:
        label_smoothing (float): label smoothing regularization coefficient
        predict_last_k (int): how many last tokens to use for the loss
            calculation, important for fast evaluation of LM perplexity
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        target_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            "target_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
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

        loss_params = {
            "label_smoothing": self.local_parameters.get("label_smoothing", 0),
            "predict_last_k": self.local_parameters.get("predict_last_k", 0),
        }
        self._loss_fn = SmoothedCrossEntropyLoss(**loss_params)
        self._pad_id = self.local_parameters['pad_id']

    def _loss_function(self, logits, target_ids):
        target_mask = mask_padded_tokens(target_ids, self._pad_id).to(logits.dtype)
        loss = self._loss_fn(logits, target_ids, target_mask)
        return loss