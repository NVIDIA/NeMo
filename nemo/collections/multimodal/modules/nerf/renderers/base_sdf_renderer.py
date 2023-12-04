import torch

from nemo.collections.multimodal.modules.renderer.base_renderer import RendererBase


class BaseSDFRenderer(RendererBase):
    def __init__(self, bound):
        super().__init__(bound)

    # TODO(ahmadki): needs a rework
    @torch.no_grad()
    def get_vertices_and_triangles(self, resolution=None, S=128):
        deform = torch.tanh(self.deform) / self.grid_size

        vertices, triangles = self.dmtet(self.verts + deform, self.sdf, self.indices)

        vertices = vertices.detach().cpu().numpy()
        triangles = triangles.detach().cpu().numpy()

        return vertices, triangles
