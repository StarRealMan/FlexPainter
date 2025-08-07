import imageio
import numpy as np
import torch
import os
from spuv.nvdiffrast_utils import render_rgb_from_texture_mesh
from spuv.ops import get_projection_matrix, get_mvp_matrix
from spuv.camera import get_c2w
from torchvision.utils import save_image

def gen_camera_path(frame_num, ele, camera_dist, fovy):
    eles = torch.tensor([ele for _ in range(frame_num)])
    azims = torch.tensor([i/frame_num * 360 for i in range(frame_num)])

    if camera_dist.shape[0] == 1:
        camera_dist = torch.tensor([camera_dist for _ in range(frame_num)])
    else:
        camera_dist = torch.tensor([camera_dist[0] for _ in range(frame_num)])

    c2w = get_c2w(azims, eles, camera_dist)
    proj = get_projection_matrix(fovy, 1, 0.1, 1000.0)
    mvp = get_mvp_matrix(c2w, proj)

    return mvp

def gen_views(eles, azims, fovy, camera_dist):
    c2w = get_c2w(azims, eles, camera_dist)
    proj = get_projection_matrix(fovy, 1, 0.1, 1000.0)
    mvps = get_mvp_matrix(c2w, proj)

    return mvps

def render_eight_views(dist, fov, device, ctx, mesh, uv_map, render_size, bg, output_path):
    eles = torch.tensor([-10, 15, -10, 15, -10, 15, -10, 15])
    azims = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315])
    dist = dist.to('cpu')
    mvps = gen_views(eles, azims, fov, dist)
    mvps = mvps.to(device)
    bg = bg.to(device)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    uv_map = uv_map.permute(1, 2, 0)
    for i in range(len(mvps)):
        mvp = mvps[i].unsqueeze(0)
        rgb = render_rgb_from_texture_mesh(ctx, mesh, uv_map, mvp, render_size, render_size, bg)
        save_image(rgb.permute(0, 3, 1, 2), os.path.join(output_path, f'{i}.png'))

def compose_video(images, output_path):
    with imageio.get_writer(output_path, fps=30, codec="libx264", quality=10) as writer:
        for image in images:
            image = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(image)

def render_video(frame_num, ele, dist, fov, device, ctx, mesh, uv_map, render_size, bg, output_path):
    mvps = gen_camera_path(frame_num, ele, dist, fov)
    mvps = mvps.to(device)
    bg = bg.to(device)
    
    uv_map = uv_map.permute(1, 2, 0)
    rgbs = []
    for i in range(len(mvps)):
        mvp = mvps[i].unsqueeze(0)
        rgb = render_rgb_from_texture_mesh(ctx, mesh, uv_map, mvp, render_size, render_size, bg)
        rgbs.append(rgb)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    compose_video(rgbs, output_path)