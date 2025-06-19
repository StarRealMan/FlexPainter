import os
import math
import random
import json
import sys
sys.path.append('.')

import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from torchvision.utils import save_image
from diffusers import FluxPriorReduxPipeline

from custom_pipe.flux_sync_cfg import FluxControlSyncCFGpipeline
from custom_pipe.flux_sync_cfg import calculate_shift, retrieve_timesteps

from spuv.ops import get_projection_matrix, get_mvp_matrix
from spuv.camera import get_c2w
from spuv.mesh_utils import load_mesh_and_uv, vertex_transform
from spuv.rasterize import NVDiffRasterizerContext
from model.utils.feature_baking import bake_image_feature_to_uv
from spuv.nvdiffrast_utils import (
    render_rgb_from_texture_mesh,
    render_xyz_from_mesh,
    render_normal_from_mesh,
    rasterize_geometry_maps,
)
from utils.pipe import mv_sync_cfg_intermediate
from utils.renderer import (
    position_to_depth,
    normalize_depth,
    generate_ray_image,
    rotate_c2w,
)

camera_poses = [(15.0, 0.0), (-15.0, 90.0), (15.0, 180.0), (-15.0, 270)]
camera_dist = 4
fovy = math.radians(30)
bg_color = [1.0, 1.0, 1.0]

base_path = "/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/3d_dataset/texture3d_obj_with_one_texture"
result_path = "./gen_data"
blank_path = "./data/blank"
if not os.path.exists(result_path):
    os.makedirs(result_path)

fix_prompt = 'a grid of 2x2 multi-view image. white background.'
obj_list = json.load(open("./json_list/obj_list.json"))
gen_option = "image"
front_dict = {}
for obj in obj_list:
    scene_name = obj['path'].split('/')[-1]
    front_dict[scene_name] = obj['front']

image_seq_len = 4096
front_disturb = 20
texture_size = 1024
resolution = 512
redux_strength = 1.0
cfg_scale = 6
true_cfg = 1
use_custom_timestep = True

@torch.no_grad()
def sample_steps(image_seq_len, scheduler, step_num, device):
    sigmas = np.linspace(1.0, 1 / step_num, step_num)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(scheduler, step_num, device, sigmas=sigmas, mu=mu)
    timesteps = torch.cat([timesteps, torch.tensor((0,), device=device)])
    random_values = torch.empty(timesteps.size(0) - 2).uniform_(0, 1)
    random_values = random_values.to(device) 
    interpolated_values = timesteps[:-2] + random_values * (timesteps[2:] - timesteps[:-2])
    costom_timesteps = torch.cat((timesteps[:1], interpolated_values, timesteps[-1:]), dim=0)
    
    return costom_timesteps

@torch.no_grad()
def gen_data_process(sub_list, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    ctx = NVDiffRasterizerContext('cuda', device)
    flux_pipe = FluxControlSyncCFGpipeline.from_pretrained('black-forest-labs/FLUX.1-Depth-dev', torch_dtype=torch.bfloat16)
    flux_pipe.load_lora_weights('./lora/lora-final.safetensors')
    flux_pipe.to(device)
    redux_pipe = FluxPriorReduxPipeline.from_pretrained('black-forest-labs/FLUX.1-Redux-dev', torch_dtype=torch.bfloat16)
    redux_pipe.to(device)

    blank_txt_path = os.path.join(blank_path, "t5.pt")
    blank_vec_path = os.path.join(blank_path, "clip.pt")
    blank_txt = torch.load(blank_txt_path, map_location=device).to(torch.bfloat16)
    blank_vec = torch.load(blank_vec_path, map_location=device).to(torch.bfloat16)

    for scene_name in sub_list:
        mesh_path = os.path.join(base_path, scene_name, scene_name + '.obj')
        uv_path = os.path.join(base_path, scene_name, "final_texture.png")
        mesh, uv_map = load_mesh_and_uv(mesh_path, uv_path, device)
        mesh = vertex_transform(mesh, mesh_scale=0.9)

        step_num = 30
        timesteps = sample_steps(image_seq_len, flux_pipe.scheduler, step_num, device)
        rand_azim = random.uniform(0, 360)

        eles = torch.tensor([pose[0] for pose in camera_poses], device=device)
        azims = torch.tensor([(pose[1] + rand_azim) % 360 for pose in camera_poses], device=device)
        camera_dist_round = torch.tensor(camera_dist, device=device).repeat(len(eles))
        c2w = get_c2w(azims, eles, camera_dist_round)
        proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(device)
        mvp = get_mvp_matrix(c2w, proj)
        bg = torch.tensor(bg_color, device=device)

        front = front_dict[scene_name]
        if front is None:
            front = random.randint(0, 3)
        front_angle = front * 90
        front_elev = random.uniform(-front_disturb, front_disturb)
        front_azim = random.uniform(-front_disturb, front_disturb) + front_angle
        camera_dist_cond = torch.tensor(camera_dist, device=device).unsqueeze(0)
        front_c2w = get_c2w(torch.tensor([front_azim], device=device), torch.tensor([front_elev], device=device), camera_dist_cond)
        front_mvp = get_mvp_matrix(front_c2w, proj)

        image_prompt = render_rgb_from_texture_mesh(ctx, mesh, uv_map[..., :3], front_mvp, resolution, resolution, bg)
        image = render_rgb_from_texture_mesh(ctx, mesh, uv_map[..., :3], mvp, resolution, resolution, bg)
        xyz, mask = render_xyz_from_mesh(ctx, mesh, mvp, resolution, resolution)
        depth = position_to_depth(xyz, c2w)
        inv_depth = normalize_depth(depth, mask).permute(0, 3, 1, 2)

        uv_position, uv_normal, uv_mask = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
        if uv_mask.sum() == 0:
            print(f"Scene {scene_name} is corrupted")
            continue
        # normal = render_normal_from_mesh(ctx, mesh, mvp, resolution, resolution)
        # rays_d = generate_ray_image(mvp, resolution, resolution)
        # rays_d = rotate_c2w(rays_d)
        # score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
        # score = torch.abs(score)

        image_prompt = (image_prompt * 255).to(torch.uint8)
        image_prompt = image_prompt.permute(0, 3, 1, 2)

        image = image.permute(0, 3, 1, 2)
        image = image.clamp(0, 1)

        images, ts = mv_sync_cfg_intermediate(
            flux_pipe, redux_pipe, fix_prompt, image_prompt, inv_depth, timesteps, use_custom_timestep,
            cfg_scale, generator, redux_strength, true_cfg, blank_txt, blank_vec
        )
        images = images.to(torch.float32)

        scene_path = os.path.join(result_path, scene_name)
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        torch.save(mvp, os.path.join(scene_path, 'mvp.pt'))
        
        for timestep in range(images.shape[0]):
            # if timestep == 0:
            #     feature = torch.cat([images[timestep].permute(0, 2, 3, 1), rays_d, score], dim=-1)
            # else:
            #     feature = images[timestep].permute(0, 2, 3, 1)
            
            # image_info = {
            #     "mvp_mtx": mvp.unsqueeze(0),
            #     "rgb": feature.unsqueeze(0)
            # }
            # uv_bakes, uv_bake_masks = bake_image_feature_to_uv(
            #     ctx, [mesh], image_info, uv_position
            # )
            # uv_bakes = uv_bakes.view(-1, feature.shape[-1], texture_size, texture_size)
            # uv_bake_masks = uv_bake_masks.view(-1, 1, texture_size, texture_size)
            scene_path_now = os.path.join(scene_path, str(int(ts[timestep])))
            if not os.path.exists(scene_path_now):
                os.makedirs(scene_path_now)
            for i in range(images.shape[1]):
                save_image(images[timestep][i], os.path.join(scene_path_now, f'rgb_{i}.png'))
                # save_image(uv_bakes[i, :3], os.path.join(scene_path_now, f'uv_bakes_{i}.png'))

                # if timestep == 0:
                #     save_image((uv_bakes[i, 3:6] + 1 ) / 2, os.path.join(scene_path, f'uv_bake_rays_{i}.png'))
                #     save_image(uv_bakes[i, 6:7], os.path.join(scene_path, f'uv_bake_scores_{i}.png'))
                #     save_image(uv_bake_masks[i], os.path.join(scene_path, f'uv_bake_masks_{i}.png'))

        # uv_position = (uv_position + 1 ) / 2
        # uv_normal = (uv_normal + 1 ) / 2
        # save_image(uv_position.permute(0, 3, 1, 2), os.path.join(scene_path, 'uv_position.png'))
        # save_image(uv_normal.permute(0, 3, 1, 2), os.path.join(scene_path, 'uv_normal.png'))
        # save_image(uv_mask.float().permute(0, 3, 1, 2), os.path.join(scene_path, 'uv_mask.png'))
        # save_image(uv_map.permute(2, 0, 1).unsqueeze(0), os.path.join(scene_path, 'uv_gt.png'))

def distribute_tasks(scene_list, num_gpus):
    sublists = [scene_list[i::num_gpus] for i in range(num_gpus)]
    processes = []
    for i in range(num_gpus):
        device = f'cuda:{i}'
        p = Process(target=gen_data_process, args=(sublists[i], device, 42))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    scene_list = os.listdir(base_path)

    filtered_scene_list = []
    for scene_name in scene_list:
        decimal_scene_name = hash(scene_name)
        if decimal_scene_name % 5 in [4]:
            filtered_scene_list.append(scene_name)
    set_start_method("spawn")
    num_gpus = torch.cuda.device_count()
    distribute_tasks(filtered_scene_list, num_gpus)

    # gen_data_process(scene_list, 'cuda:0', 42)