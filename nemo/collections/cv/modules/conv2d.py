from torch import nn
from torch.nn.common_types import _size_2_t
from typing import Union
from torch.nn import init

class Conv2d(nn.Conv2d):

    """
    Conv2d layer with ResNet initialization:

    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,

        weight_init: str = "default",
        bias_init: str = "default"
    ):
        
        super(Conv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Weight Init
        assert weight_init in ["default", "he_normal"]
        if weight_init == "he_normal":
            init.kaiming_normal_(self.weight)

        # Bias Init
        assert bias_init in ["default", "zeros"]
        if self.bias is not None:
            if bias_init == "zeros":
                init.zeros_(self.bias)