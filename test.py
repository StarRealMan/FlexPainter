import os
import math
import torch
from torchvision.utils import save_image
from diffusers import FluxPriorReduxPipeline

from pipeline.outpainter import OutpainterPipe
from pipeline.flux_sync_cfg import FluxControlSyncCFGpipeline

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
from pipeline.weighter import Weighter
from utils.config import config
from utils.video import render_video
from utils.misc import process_image
from utils.pipe import mv_sync_cfg_generation
from utils.voronoi import voronoi_solve
from utils.renderer import (
    position_to_depth,
    normalize_depth,
    generate_ray_image,
    rotate_c2w,
)

if __name__ == '__main__':
    args = config()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    ctx = NVDiffRasterizerContext('cuda', args.device)
    camera_poses = [(15.0, 0.0), (-15.0, 90.0), (15.0, 180.0), (-15.0, 270)]
    camera_dist = 20 / 9
    fovy = math.radians(30)

    flux_pipe = FluxControlSyncCFGpipeline.from_pretrained(args.base_model, torch_dtype=args.dtype)
    if args.lora_model is not None:
        flux_pipe.load_lora_weights(args.lora_model)
    flux_pipe.to(args.device)
    redux_pipe = FluxPriorReduxPipeline.from_pretrained(args.redux_model, torch_dtype=args.dtype)
    redux_pipe = redux_pipe.to(args.device)

    mesh = load_mesh_only(args.mesh_path, args.device)
    mesh = vertex_transform(mesh, mesh_scale=0.5)

    eles = torch.tensor([pose[0] for pose in camera_poses], device=args.device)
    azims = torch.tensor([(pose[1] + args.render_azim) % 360 for pose in camera_poses], device=args.device)
    camera_dist = torch.tensor(camera_dist, device=args.device).repeat(len(eles))
    c2w = get_c2w(azims, eles, camera_dist)
    proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(args.device)
    mvp = get_mvp_matrix(c2w, proj)

    # blank_txt_path = os.path.join(args.blank_path, "t5.pt")
    # blank_vec_path = os.path.join(args.blank_path, "clip.pt")
    blank_txt_path = os.path.join(args.blank_path, "redux_t5.pt")
    blank_vec_path = os.path.join(args.blank_path, "redux_clip.pt")
    blank_txt = torch.load(blank_txt_path, map_location=args.device).to(args.dtype)
    blank_vec = torch.load(blank_vec_path, map_location=args.device).to(args.dtype)

    outpainter_pipe = OutpainterPipe(args.device, args.dtype)
    outpainter_pipe.load_weights(args.outpainter_model)
    weighter = Weighter(args.texture_size, args.resolution, args.device)
    weighter.load_weights(args.weighter_model)

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
        weighter.preprocess(renderer)
        images = mv_sync_cfg_generation(
            flux_pipe, redux_pipe, args.prompt, args.image_prompt, inv_depth, 
            args.sample_steps, args.cfg_scale, generator, args.image_strength, args.true_cfg,
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

        uv_pred = weighter(uv_bakes[:, :3], uv_bake_masks, torch.tensor([0]).to(args.device))
        uv_position = uv_position.permute(0, 3, 1, 2)
        uv_mask = uv_mask.float().permute(0, 3, 1, 2)
        uv_pred = uv_pred * final_mask

        image_final_mask = torch.bitwise_xor(images_white[:, :1].bool(), mask.permute(0, 3, 1, 2).bool()).float()
        images = images * image_final_mask

        final_res = outpainter_pipe(
            [mesh], args.outpainter_prompt, images, uv_pred, final_mask, uv_mask, uv_position,
            args.outpainter_sample_steps, args.outpainter_cfg_scale, (0.0, 1.0), 0.0, 
        )

        final_res = voronoi_solve(final_res.squeeze(0).permute(1, 2, 0), uv_mask.squeeze(), device=args.device)
        final_res = final_res.permute(2, 0, 1).unsqueeze(0)

        for i in range(len(images)):
            save_image(images[i], os.path.join(args.result_path, f'rgb_{i}.png'))
            save_image(uv_bakes[i, :3], os.path.join(args.result_path, f'uv_bakes_{i}.png'))
        
        save_image(uv_pred, os.path.join(args.result_path, 'uv_pred.png'))
        save_image(final_res, os.path.join(args.result_path, 'uv_paint.png'))
        save_image(final_mask, os.path.join(args.result_path, 'final_mask.png'))
        render_video(args.frame_num, args.render_ele, camera_dist, fovy, args.device, 
                    ctx, mesh, final_res.squeeze(0), args.resolution, torch.tensor([0, 0, 0]), 
                    os.path.join(args.result_path, 'video.mp4'))
        