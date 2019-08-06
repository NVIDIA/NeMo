# noinspection PyPep8Naming

from torch import nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import NeuralType, AxisType, BatchTag, TimeTag, \
    ChannelTag
from nemo_asr.jasper import init_weights as jasper_init_weights


class JasperRNNConnector(TrainableNM):
    """Connector between jasper encoder and some other module, that does
    change number of channels.

    Args:
        in_channels: Number of channels of input tensor
        out_channels: Number of channels to reshape to

    """

    @staticmethod
    def create_ports():
        input_ports = {
            'tensor': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag),
                2: AxisType(TimeTag)
            })
        }
        output_ports = {
            'tensor': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)

        self.icnn = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)

        self.apply(jasper_init_weights)
        self.to(self._device)

    def forward(self, tensor):
        # tensor = F.relu(tensor)
        # tensor = F.dropout(tensor, 0.2)
        tensor = self.icnn(tensor)
        tensor = self.bn(tensor)
        tensor = tensor.transpose(1, 2)
        return tensor
