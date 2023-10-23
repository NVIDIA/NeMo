import torch
import torch.nn as nn

# TODO(ahmadki): abstract class
class NeRFBackgroundBase(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, rays_d: torch.Tensor) -> torch.Tensor:
        """
        positions = [B*N, 3]
        """
        raise NotImplementedError

    def forward_net(self, rays_d_encoding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        rays_d_encoding = self.encode(rays_d)
        features = self.forward_net(rays_d_encoding)
        features = torch.sigmoid(features)
        return features
