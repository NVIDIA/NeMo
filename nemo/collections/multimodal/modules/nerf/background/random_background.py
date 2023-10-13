import random
from typing import Tuple

import torch
import torch.nn as nn


class RandomBackground(nn.Module):
    def __init__(self, base_background: Tuple, random_ratio: float) -> None:
        super().__init__()
        self.random_ratio = random_ratio
        self.num_output_dims = len(base_background)
        self.register_buffer("base_background", torch.tensor(base_background))

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        if random.random() < self.random_ratio:
            return torch.rand(rays_d.shape[0], self.num_output_dims).to(rays_d)
        else:
            return self.base_background.to(rays_d).expand(rays_d.shape[0], -1)
