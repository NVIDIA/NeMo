# # -*- coding: utf-8 -*-

# # Copyright 2020 Tomoki Hayashi
# #  MIT License (https://opensource.org/licenses/MIT)

# """MelGAN Modules."""

# import logging

# import numpy as np
# import torch


# class GANTTSDiscriminator(torch.nn.Module):
#     def __init__(
#         self, in_channels=1, out_channels=1, input_lengths=240,
#     ):
#     super().__init__()
#     self.downsample_


# # class GANTTSMultiDiscriminator(torch.nn.Module):
# #     """MelGAN discriminator module."""

# #     def __init__(
# #         self, in_channels=1, out_channels=1, input_lengths=[240, 480, 960, 1920, 3600],
# #     ):
# #         super().__init__()
# #         self.discriminators = torch.nn.ModuleList()
# #         # add discriminators
# #         for _ in range(scales):
# #             self.discriminators += [
# #                 MelGANDiscriminator(
# #                     in_channels=in_channels,
# #                     out_channels=out_channels,
# #                     kernel_sizes=kernel_sizes,
# #                     channels=channels,
# #                     max_downsample_channels=max_downsample_channels,
# #                     bias=bias,
# #                     downsample_scales=downsample_scales,
# #                     nonlinear_activation=nonlinear_activation,
# #                     nonlinear_activation_params=nonlinear_activation_params,
# #                     pad=pad,
# #                     pad_params=pad_params,
# #                 )
# #             ]

# #         # check kernel size is valid
# #         assert len(kernel_sizes) == 2
# #         assert kernel_sizes[0] % 2 == 1
# #         assert kernel_sizes[1] % 2 == 1

# #         # add first layer
# #         self.layers += [
# #             torch.nn.Sequential(
# #                 getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
# #                 torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
# #                 getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
# #             )
# #         ]

# #         # add downsample layers
# #         in_chs = channels
# #         for downsample_scale in downsample_scales:
# #             out_chs = min(in_chs * downsample_scale, max_downsample_channels)
# #             self.layers += [
# #                 torch.nn.Sequential(
# #                     torch.nn.Conv1d(
# #                         in_chs,
# #                         out_chs,
# #                         kernel_size=downsample_scale * 10 + 1,
# #                         stride=downsample_scale,
# #                         padding=downsample_scale * 5,
# #                         groups=in_chs // 4,
# #                         bias=bias,
# #                     ),
# #                     getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
# #                 )
# #             ]
# #             in_chs = out_chs

# #         # add final layers
# #         out_chs = min(in_chs * 2, max_downsample_channels)
# #         self.layers += [
# #             torch.nn.Sequential(
# #                 torch.nn.Conv1d(in_chs, out_chs, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2, bias=bias,),
# #                 getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
# #             )
# #         ]
# #         self.layers += [
# #             torch.nn.Conv1d(out_chs, out_channels, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2, bias=bias,),
# #         ]

# #     def forward(self, x):
# #         """Calculate forward propagation.

# #         Args:
# #             x (Tensor): Input noise signal (B, 1, T).

# #         Returns:
# #             List: List of output tensors of each layer.

# #         """
# #         outs = []
# #         for f in self.layers:
# #             x = f(x)
# #             outs += [x]

# #         return outs
