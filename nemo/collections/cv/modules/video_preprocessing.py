import torch
import torchvision
from torch import nn

from nemo.collections.cv.modules import Permute
from nemo.core.classes import NeuralModule, typecheck


class VideoPreprocessor(NeuralModule):

    """ Lip Reading Video Pre-processing

    args:
        grayscale: convert images to grayscale
        normalize: normalize videos
        resize: resize videos
        resize_size: output image size for resize
        norm_mean: normalize mean
        norm_std: normalize std
    
    """

    def __init__(self, grayscale, normalize, resize, resize_size, norm_mean, norm_std):
        super().__init__()

        # Params
        self.grayscale = grayscale
        self.normalize = normalize
        self.resize = resize
        self.resize_size = resize_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.transforms = nn.ModuleList()

        # Convert float32 [0:255] -> [0:1]
        self.transforms.append(torchvision.transforms.ConvertImageDtype(dtype=torch.float32))

        # Convert Channels First
        self.transforms.append(Permute(dims=(0, 4, 1, 2, 3)))  # (B, T, H, W, C) -> (B, C, T, H, W)

        # Resize
        if self.resize:
            self.transforms.append(ResizeVideo(self.resize_size))  # (B, C, T, H, W) -> (B, C, T, H', W')

        # Grayscale
        if self.grayscale:
            self.transforms.append(
                nn.Sequential(
                    Permute(dims=(0, 2, 1, 3, 4)),  # (B, C, T, H, W) -> (B, T, C, H, W)
                    torchvision.transforms.Grayscale(),
                    Permute(dims=(0, 2, 1, 3, 4)),  # (B, T, C, H, W) -> (B, C, T, H, W)
                )
            )

        # Normalize
        if self.normalize:
            self.transforms.append(NormalizeVideo(mean=norm_mean, std=norm_std))

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):

        for transform in self.transforms:
            input_signal = transform(input_signal)

        return input_signal, length


class NormalizeVideo(NeuralModule):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).reshape(len(mean), 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).reshape(len(std), 1, 1, 1), persistent=False
        )

    def forward(self, x):

        x = (x - self.mean) / self.std

        return x


class ResizeVideo(NeuralModule):
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.resize = torchvision.transforms.Resize(size=self.size)

    def forward(self, x):

        # (B, C, T, H, W)
        if x.dim() == 5:

            B, C = x.shape[:2]
            x = x.flatten(start_dim=0, end_dim=1)
            x = self.resize(x)
            x = x.reshape((B, C) + x.shape[1:])

        # (C, T, H, W)
        elif x.dim() == 4:
            x = self.resize(x)

        return x
