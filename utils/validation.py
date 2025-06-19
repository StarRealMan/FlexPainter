import os
import torch

from utils.renderer import data_process
from spuv.mesh_utils import load_mesh_and_uv, vertex_transform, load_image
from utils.misc import to_device

rgb_length = 4
def load_validation(ctx, sample_path, mesh_scale, bg):
    device = ctx.device
    samples = []
    for scene in os.listdir(sample_path):
        scene_path = os.path.join(sample_path, scene)
        mesh_path = os.path.join(scene_path, "model.obj")
        uv_path = os.path.join(scene_path, "model.png")

        mesh, uv_map = load_mesh_and_uv(mesh_path, uv_path, "cpu")
        mesh = vertex_transform(mesh, mesh_scale = mesh_scale)
        mesh = to_device(mesh, device)
        uv_map = uv_map.to(device)
        mvp = torch.load(os.path.join(scene_path, "mvp.pt"))

        for timestep in os.listdir(os.path.join(scene_path, "rgb")):
            gen_rgbs = []
            for rgb_num in range(rgb_length):
                gen_rgb_path = os.path.join(scene_path, "rgb", f"{timestep}", f"image_{rgb_num}.png")
                gen_rgb = load_image(gen_rgb_path)
                gen_rgb = torch.from_numpy(gen_rgb)
                gen_rgbs.append(gen_rgb.to(torch.float32))
            gen_rgbs = torch.stack(gen_rgbs, dim=0).to(device)

            uv_position, uv_normal, uv_mask, uv_bakes, uv_bake_masks, rgbs = data_process(
                ctx, [mesh], [uv_map], [mvp], [gen_rgbs], bg
            )

            uv_gen_bakes = uv_bakes[:, :, 3:6]
            uv_rays = uv_bakes[:, :, 6:9]
            uv_scores = uv_bakes[:, :, 9:10]
            uv_bakes = uv_bakes[:, :, :3]

            sample = {
                "gen_bakes": uv_gen_bakes,
                "rays": uv_rays,
                "ref_scores": uv_scores,
                "position_map": uv_position,
                "normal_map": uv_normal,
                "timestep": timestep,
                "mesh": mesh,
                "uv_bake_masks": uv_bake_masks,
            }
            samples.append(sample)
        
    return samples
