from typing import Tuple

import torch
import torch.nn as nn


class StaticBackground(nn.Module):
    def __init__(self, background: Tuple) -> None:
        super().__init__()
        self.register_buffer("background", torch.tensor(background))

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        background = self.background.to(rays_d)
        return background.expand(rays_d.shape[0], -1)
