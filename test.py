import os
import math
import random
import argparse
from PIL import Image

import torch
from torchvision.utils import save_image

from custom_pipe.flux_sync_cfg import FluxControlSyncCFGpipeline
from diffusers import FluxPriorReduxPipeline

from spuv.ops import get_projection_matrix, get_mvp_matrix
from spuv.camera import get_c2w
from spuv.mesh_utils import load_mesh_only, vertex_transform
from spuv.nvdiffrast_utils import render_xyz_from_mesh, rasterize_geometry_maps
from spuv.rasterize import NVDiffRasterizerContext
from model.utils.feature_baking import bake_image_feature_to_uv
from spuv.nvdiffrast_utils import (
    render_xyz_from_mesh,
    render_normal_from_mesh,
    rasterize_geometry_maps,
)
from custom_pipe.outpainter import OutpainterPipe
from utils.weighter import Cos_Weighter
from utils.renderer import position_to_depth, normalize_depth
from utils.video import render_video
from utils.pipe import mv_sync_cfg_generation
from utils.renderer import (
    position_to_depth,
    normalize_depth,
    generate_ray_image,
    rotate_c2w,
)
import cv2
import numpy as np

def process_image(image):
    image = np.transpose(image, (1, 2, 0))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    grad_magnitude = np.uint8(grad_magnitude)

    edge_threshold = 30
    edges = cv2.threshold(grad_magnitude, edge_threshold, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    mask = np.zeros_like(gray)

    mask[dilated_edges > 0] = 255

    result = np.zeros_like(image)

    white_threshold = 240
    for i in range(3): 
        result[:, :, i] = np.where((gray > white_threshold) & (dilated_edges > 0), 255, 0)
    
    return result



outpainter_config = {
    "in_channels": 10,
    "out_channels": 3,
    "num_layers": [1, 1, 1, 1, 1],
    "point_block_num": [1, 1, 2, 4, 6],
    "block_out_channels": [32, 256, 1024, 1024, 2048],
    "dropout": [0.0, 0.0, 0.0, 0.1, 0.1],
    "use_uv_head": True,
    "block_type": ["uv", "point_uv", "uv_dit", "uv_dit", "uv_dit"],
    "voxel_size": [0.01, 0.02, 0.05, 0.05, 0.05],
    "window_size": [0, 256, 256, 512, 1024],
    "num_heads": [4, 4, 16, 16, 16],
    "skip_input": True,
    "skip_type": "adaptive",
    "weights": None
}

clip_config = {
    "pretrained_model_name_or_path": "lambdalabs/sd-image-variations-diffusers"
}

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='./demo/bird/model.obj')
    parser.add_argument('--blank_path', type=str, default='./data/blank')
    parser.add_argument('--base_model', type=str, default='black-forest-labs/FLUX.1-Depth-dev')
    parser.add_argument('--redux_model', type=str, default='black-forest-labs/FLUX.1-Redux-dev')
    parser.add_argument('--outpainter_model', type=str, default='./ckpts/outpainter/texgen_v1.ckpt')
    parser.add_argument('--lora_model', type=str, default="./ckpts/lora/lora-final.safetensors")
    parser.add_argument('--result_path', type=str, default='./test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--image_prompt', type=str, default=None)
    parser.add_argument('--stylize', action='store_true', default=False)
    parser.add_argument('--sample_steps', type=int, default=30)
    parser.add_argument('--mixing_step', type=int, default=10)
    parser.add_argument('--cfg_scale', type=float, default=6.0)
    parser.add_argument('--true_cfg', type=float, default=1.0)
    parser.add_argument('--redux_strength', type=float, default=0.3)
    parser.add_argument('--render_azim', type=float, default=-1)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--texture_size', type=int, default=1024)
    parser.add_argument('--outpainter_sample_steps', type=int, default=30)
    parser.add_argument('--outpainter_cfg_scale', type=float, default=3.5)
    parser.add_argument('--frame_num', type=int, default=90)
    parser.add_argument('--render_ele', type=float, default=15.0)

    args = parser.parse_args()

    fix_prompt = 'a grid of 2x2 multi-view image. white background.'
    outpainter_default_prompt = 'a 3D model'
    if args.prompt is not None:
        args.outpainter_prompt = args.prompt
        args.prompt = fix_prompt + ' ' + args.prompt
    else:
        args.outpainter_prompt = outpainter_default_prompt
        args.prompt = fix_prompt

    if args.image_prompt is not None:
        args.image_prompt = Image.open(args.image_prompt).convert('RGB')
    
    if args.prompt is None and args.image_prompt is None:
        raise ValueError('Please provide either a prompt or an image prompt.')
    
    if args.render_azim < 0 or args.render_azim >= 360:
        args.render_azim = random.uniform(0, 360)

    if args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16
    elif args.dtype == 'float32':
        args.dtype = torch.float32

    args.result_path = os.path.join(args.result_path, args.mesh_path.split('/')[-2])
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    return args

if __name__ == '__main__':
    args = config()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    ctx = NVDiffRasterizerContext('cuda', args.device)
    camera_poses = [(15.0, 0.0), (-15.0, 90.0), (15.0, 180.0), (-15.0, 270)]
    camera_dist = 4
    fovy = math.radians(30)

    flux_pipe = FluxControlSyncCFGpipeline.from_pretrained(args.base_model, torch_dtype=args.dtype)
    if args.lora_model is not None:
        flux_pipe.load_lora_weights(args.lora_model)
    flux_pipe.to(args.device)
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(args.redux_model, torch_dtype=args.dtype)
    redux_pipe = redux_pipe.to(args.device)

    mesh = load_mesh_only(args.mesh_path, args.device)
    mesh = vertex_transform(mesh, mesh_scale=0.9)

    eles = torch.tensor([pose[0] for pose in camera_poses], device=args.device)
    azims = torch.tensor([(pose[1] + args.render_azim) % 360 for pose in camera_poses], device=args.device)
    camera_dist = torch.tensor(camera_dist, device=args.device).repeat(len(eles))
    c2w = get_c2w(azims, eles, camera_dist)
    proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(args.device)
    mvp = get_mvp_matrix(c2w, proj)

    blank_txt_path = os.path.join(args.blank_path, "t5.pt")
    blank_vec_path = os.path.join(args.blank_path, "clip.pt")
    blank_txt = torch.load(blank_txt_path, map_location=args.device).to(args.dtype)
    blank_vec = torch.load(blank_vec_path, map_location=args.device).to(args.dtype)

    outpainter_pipe = OutpainterPipe(outpainter_config, clip_config, args.device, args.dtype)
    outpainter_pipe.load_weights(args.outpainter_model)

    with torch.no_grad():
        uv_position, uv_normals, uv_mask = rasterize_geometry_maps(ctx, mesh, args.texture_size, args.texture_size)
        xyz, mask = render_xyz_from_mesh(ctx, mesh, mvp, args.resolution, args.resolution)
        depth = position_to_depth(xyz, c2w)
        inv_depth = normalize_depth(depth, mask).permute(0, 3, 1, 2)

    renderer = {
        "ctx": ctx,
        "mesh": mesh,
        "mvps": mvp,
    }
    
    weighter = Cos_Weighter(renderer, args.texture_size, args.resolution).eval()

    images = mv_sync_cfg_generation(
        flux_pipe, redux_pipe, args.prompt, args.image_prompt, inv_depth, 
        args.sample_steps, args.cfg_scale, generator, args.redux_strength, args.true_cfg,
        args.texture_size, args.texture_size, args.mixing_step, renderer,
        blank_txt, blank_vec, weighter, args.stylize
    )
    images = images.to(torch.float32)
    
    images_white = []
    for i in range(len(images)):
        images[i] = images[i] * mask[i].permute(2, 0, 1)
        images_white.append(process_image((images[i] * 255.0).cpu().numpy()))
        images_white[i] = torch.tensor((images_white[i] / 255.0)).permute(2, 0, 1)
    
    images_white = torch.stack(images_white).to(device=args.device)
    with torch.no_grad():
        normal = render_normal_from_mesh(ctx, mesh, mvp, args.resolution, args.resolution)
        rays_d = generate_ray_image(mvp, args.resolution, args.resolution)
        rays_d = rotate_c2w(rays_d)
        score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
        score = torch.abs(score)
        
        feature = torch.cat([images.permute(0, 2, 3, 1), rays_d, score, images_white.permute(0, 2, 3, 1)], dim=-1)
        uv_position, uv_normal, uv_mask = rasterize_geometry_maps(ctx, mesh, args.texture_size, args.texture_size)
        image_info = {
            "mvp_mtx": mvp.unsqueeze(0),
            "rgb": feature.unsqueeze(0)
        }
        uv_bakes, uv_bake_masks = bake_image_feature_to_uv(
            ctx, [mesh], image_info, uv_position
        )
        uv_bakes = uv_bakes.view(-1, feature.shape[-1], args.texture_size, args.texture_size)
        uv_bake_masks = uv_bake_masks.view(-1, 1, args.texture_size, args.texture_size)
        uv_bake_mask = uv_bake_masks.sum(dim=0, keepdim=True) > 0
        
        uv_bakes_white_masks = (uv_bakes[:, 7:] != 0).any(dim=1, keepdim=True).float()
        uv_bake_white_mask = uv_bakes_white_masks.sum(dim=0, keepdim=True) > 0
        final_mask = torch.bitwise_xor(uv_bake_white_mask, uv_bake_mask).float()

    if weighter is None:
        uv_pred = (uv_bakes[:, :3] * uv_bake_masks).sum(dim=0) / (uv_bake_masks.sum(dim=0) + 1e-6)
    else:
        uv_pred = weighter(uv_bakes, uv_bake_masks, 0)

    uv_position = uv_position.permute(0, 3, 1, 2)
    uv_mask = uv_mask.float().permute(0, 3, 1, 2)
    uv_pred = uv_pred.unsqueeze(0)
    final_res = outpainter_pipe(
        [mesh], args.outpainter_prompt, images, uv_pred, final_mask, uv_mask, uv_position,
        args.outpainter_sample_steps, args.outpainter_cfg_scale, (0.0, 1.0), 0.0, 
    )

    for i in range(len(images)):
        save_image(images[i] * mask[i].permute(2, 0, 1), os.path.join(args.result_path, f'rgb_{i}.png'))
    save_image(uv_pred, os.path.join(args.result_path, 'uv_pred.png'))
    save_image(final_res, os.path.join(args.result_path, 'uv_paint.png'))

    render_video(args.frame_num, args.render_ele, camera_dist, fovy, args.device, 
                 ctx, mesh, final_res.squeeze(0), args.resolution, torch.tensor([0, 0, 0]), 
                 os.path.join(args.result_path, 'video.mp4'))
