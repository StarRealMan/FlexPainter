import torch
from safetensors.torch import load_file

from model.weighter_net import WeighterNet
from spuv.nvdiffrast_utils import render_normal_from_mesh, rasterize_geometry_maps
from model.utils.feature_baking import bake_image_feature_to_uv
from utils.renderer import generate_ray_image, rotate_c2w

class Weighter():
    def __init__(self, tex_size, render_size, device):
        weighter_config = {
            "in_channels": 34,
            "out_channels": 4,
            "num_layers": [1, 1, 1],
            "point_block_num": [1, 1, 2],
            "block_out_channels": [64, 256, 1024],
            "dropout": [0.0, 0.0, 0.0],
            "use_uv_head": True,
            "block_type": ["uv", "point_uv", "uv_dit"],
            "voxel_size": [0.01, 0.02, 0.05],
            "window_size": [0, 256, 512],
            "num_heads": [4, 4, 16],
            "weights": None
        }
        self.weighternet = WeighterNet(weighter_config).to(device)
        self.render_size = render_size
        self.tex_size = tex_size

    def load_weights(self, path):
        checkpoint = load_file(path)
        self.weighternet.load_state_dict(checkpoint)
    
    def preprocess(self, renderer):
        ctx = renderer["ctx"]
        mesh = renderer["mesh"]
        mvps = renderer["mvps"]

        uv_position, uv_normal, _ = rasterize_geometry_maps(ctx, mesh, self.tex_size, self.tex_size)
        normal = render_normal_from_mesh(ctx, mesh, mvps, self.render_size, self.render_size)
        rays_d = generate_ray_image(mvps, self.render_size, self.render_size)
        rays_d = rotate_c2w(rays_d)
        score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
        score = torch.abs(score)
        feature = torch.cat([rays_d, score], dim=-1)

        image_info = {
            "mvp_mtx": mvps.unsqueeze(0),
            "rgb": feature.unsqueeze(0)
        }
        uv_bakes, _ = bake_image_feature_to_uv(
            ctx, [mesh], image_info, uv_position
        )
        uv_bakes = uv_bakes.view(-1, 4, self.tex_size, self.tex_size)
        
        self.rays = uv_bakes[:, :3]
        self.score = uv_bakes[:, -1]
        self.uv_position = uv_position.permute(0, 3, 1, 2)
        self.uv_normal = uv_normal.permute(0, 3, 1, 2)
        self.mesh = mesh

    def __call__(self, uv_bakes, uv_bake_masks, timestep):
        uv_bake, _ = self.weighternet(
            gen_bakes = uv_bakes.unsqueeze(0),
            rays = self.rays.unsqueeze(0),
            ref_scores = self.score,
            position_map = self.uv_position,
            normal_map = self.uv_position,
            timestep = timestep,
            mesh = self.mesh,
            uv_bake_masks = uv_bake_masks.unsqueeze(0),
        )

        return uv_bake
