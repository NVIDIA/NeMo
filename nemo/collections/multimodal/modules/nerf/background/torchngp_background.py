from typing import Dict

import torch

from nemo.collections.multimodal.modules.nerf.background.nerf_background_base import NeRFBackgroundBase
from nemo.collections.multimodal.modules.nerf.geometry.layers import MLP
from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.encoding import get_encoder


class TorchNGPBackground(NeRFBackgroundBase):
    def __init__(
        self, encoder_type: str, encoder_input_dims: int, encoder_multi_res: int, num_output_dims: int, net_cfg: Dict
    ):
        super().__init__()

        self.encoder, self.encoder_output_dims = get_encoder(
            encoder_type, input_dim=encoder_input_dims, multires=encoder_multi_res
        )
        self.background_net = MLP(
            num_input_dims=self.encoder_output_dims,
            num_output_dims=num_output_dims,
            num_hidden_dims=net_cfg.num_hidden_dims,
            num_layers=net_cfg.num_layers,
            bias=net_cfg.bias,
        )

    def encode(self, rays_d: torch.Tensor) -> torch.Tensor:
        return self.encoder(rays_d)

    def forward_net(self, rays_d_encoding: torch.Tensor) -> torch.Tensor:
        return self.background_net(rays_d_encoding)
