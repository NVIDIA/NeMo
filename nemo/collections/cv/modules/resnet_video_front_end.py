from nemo.core.classes.module import NeuralModule
from nemo.collections.cv.modules.resnet import ResNet
from torch import nn
import torch

class ResNetVideoFrontEnd(NeuralModule):
    """
    Lip Reading / Visual Speech Recognition (VSR) ResNet Front-End Network

    Paper:
    'Audio-Visual Efficient Conformer for Robust Speech Recognition' by Burchi and Timofte
    https://arxiv.org/abs/2301.01456

    Args:
        in_channels: number of inputs video channels, 1 for grayscale and 3 for RGB
        model: model size in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
        dim_output: output feature dimension for linear projection after spacial average pooling
        out_channels_first: Whether outputs should have channels_first format (Batch, Dout, Time) or channels_last (Batch, Time, Dout)
    """

    def __init__(self, in_channels=1, model="ResNet18", dim_output=256, out_channels_first=True):
        super(ResNetVideoFrontEnd, self).__init__()

        self.front_end = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ResNet(include_stem=False, dim_output=dim_output, model=model)
        )

        self.out_channels_first = out_channels_first

    def forward(self, input_signal):

        # Front-End Network (Batch, Din, Time, Height, Width) -> (Batch, Dout, Time)
        input_signal = self.front_end(input_signal)

        # Transpose to channels_last format (Batch, Dout, Time) -> (Batch, Time, Dout)
        if not self.out_channels_first:
            input_signal = input_signal.transpose(1, 2)

        return input_signal
