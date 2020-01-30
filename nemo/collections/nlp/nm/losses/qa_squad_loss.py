from torch import nn

from nemo.backends.pytorch import LossNM
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag, ChannelTag


class QuestionAnsweringLoss(LossNM):
    """
    Neural module which implements QuestionAnswering loss.
    Args:
        logits: Output of question answering head, which is a token classfier.
        start_positions: Ground truth start positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
        end_positions: Ground truth end positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        start_positions:
            0: AxisType(BatchTag)

        end_positions:
            0: AxisType(BatchTag)
        """
        return {
            "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            "start_positions": NeuralType({0: AxisType(BatchTag)}),
            "end_positions": NeuralType({0: AxisType(BatchTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)

        start_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        end_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "loss": NeuralType(None),
            "start_logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "end_logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)

    def _loss_function(self, **kwargs):
        logits = kwargs['logits']
        start_positions = kwargs['start_positions']
        end_positions = kwargs['end_positions']
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss, start_logits, end_logits