import torch
import torch.nn as nn

from spuv.nvdiffrast_utils import render_normal_from_mesh, rasterize_geometry_maps
from model.utils.feature_baking import bake_image_feature_to_uv
from utils.renderer import generate_ray_image, rotate_c2w

class Cos_Weighter(nn.Module):
    def __init__(self, renderer, tex_size, render_size):
        super().__init__()
        ctx = renderer["ctx"]
        mesh = renderer["mesh"]
        mvps = renderer["mvps"]

        normal = render_normal_from_mesh(ctx, mesh, mvps, render_size, render_size)
        rays_d = generate_ray_image(mvps, render_size, render_size)
        rays_d = rotate_c2w(rays_d)
        score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
        score = torch.abs(score)

        uv_position, _, _ = rasterize_geometry_maps(ctx, mesh, tex_size, tex_size)
        image_info = {
            "mvp_mtx": mvps.unsqueeze(0),
            "rgb": score.unsqueeze(0)
        }
        uv_bakes, _ = bake_image_feature_to_uv(
            ctx, [mesh], image_info, uv_position
        )
        self.weights = uv_bakes.view(-1, 1, tex_size, tex_size)

    def forward(self, uv_bakes, uv_bake_masks, timestep):
        uv_bake = (uv_bakes[:, :3] * self.weights).sum(dim=0) / (self.weights.sum(dim=0) + 1e-6)

        return uv_bake

class T3D_Weighter(Cos_Weighter):
    def __init__(self, weighter_net, renderer, tex_size, render_size):
        super().__init__(renderer, tex_size, render_size)


    def load_weights(self, path):
        self.weights = torch.load(path)

    def forward(self, uv_bakes, uv_bake_masks, timestep):
        uv_bake = (uv_bakes[:, :3] * self.weights).sum(dim=0) / (self.weights.sum(dim=0) + 1e-6)

        return uv_bake
