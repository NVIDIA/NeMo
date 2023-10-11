from nemo.collections.multimodal.modules.nerf.geometry.nerf_base import DensityActivationEnum
from nemo.collections.multimodal.modules.renderer.base_renderer import RendererBase


class BaseVolumeRenderer(RendererBase):
    def __init__(self, bound, update_interval):
        super().__init__(bound, update_interval)
