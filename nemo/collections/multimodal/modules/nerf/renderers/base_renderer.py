import torch
import torch.nn as nn

# TODO(ahmadki): make abstract
class BaseRenderer(nn.Module):
    def __init__(self, bound, update_interval):
        super().__init__()
        self.bound = bound
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)
        self.update_interval = update_interval

    @torch.no_grad()
    def update_step(self, epoch: int, global_step: int, decay: float = 0.95, **kwargs):
        raise NotImplementedError

    def forward(self, rays_o, rays_d, return_normal_image=False, return_normal_perturb=False, **kwargs):
        raise NotImplementedError
