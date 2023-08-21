import torch
from torch import nn
from torch.nn.modules.utils import _pair

from nemo.collections.cv.modules import Conv2d
from nemo.core.classes.module import NeuralModule


class ResNetBlock(NeuralModule):

    """ ResNet Residual Block used by ResNet18 and ResNet34 networks.
    References: "Deep Residual Learning for Image Recognition", He et al.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_features, out_features, kernel_size, stride, weight_init="he_normal", bias_init="zeros"):
        super(ResNetBlock, self).__init__()

        # Convert to pair
        kernel_size = _pair(kernel_size)

        # layers
        self.layers = nn.Sequential(
            Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                weight_init=weight_init,
                bias_init=bias_init,
                padding=((kernel_size[0] - 1) // 2, kernel_size[1] // 2),
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            Conv2d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                bias=False,
                weight_init=weight_init,
                bias_init=bias_init,
                padding=((kernel_size[0] - 1) // 2, kernel_size[1] // 2),
            ),
            nn.BatchNorm2d(out_features),
        )

        # Residual Block
        if torch.prod(torch.tensor(stride)) > 1 or in_features != out_features:
            self.residual = nn.Sequential(
                Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    weight_init=weight_init,
                    bias_init=bias_init,
                ),
                nn.BatchNorm2d(out_features),
            )
        else:
            self.residual = nn.Identity()

        # Joined Post Act
        self.joined_post_act = nn.ReLU()

    def forward(self, x):

        # Forward Layers
        x = self.joined_post_act(self.layers(x) + self.residual(x))

        return x
