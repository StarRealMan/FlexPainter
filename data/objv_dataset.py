import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from spuv.mesh_utils import load_image, load_mesh_and_uv, vertex_transform

class ObjaverseDataset(Dataset):
    def __init__(self, base_path, gen_path, mesh_scale=0.9, rgb_len=4, use_try_catch=True):
        self.mesh_scale = mesh_scale
        self.rgb_len = rgb_len

        self.use_try_catch = use_try_catch

        self.all_scenes = []
        for scene_name in os.listdir(gen_path):
            gen_scene = os.path.join(gen_path, scene_name)
            base_scene = os.path.join(base_path, scene_name)
            self.all_scenes.append((gen_scene, base_scene))

    def get_data(self, idx):
        gen_scene, base_scene = self.all_scenes[idx]

        timesteps = os.listdir(gen_scene)
        timestep = random.choice(timesteps)
        gen_rgbs = []
        for rgb_num in range(self.rgb_len):
            gen_rgb_path = os.path.join(gen_scene, timestep, f"rgb_{rgb_num}.png")
            gen_rgb = load_image(gen_rgb_path).astype(np.float32)
            gen_rgbs.append(gen_rgb)
        gen_rgbs = np.stack(gen_rgbs, axis=0)
        mvps = torch.load(os.path.join(gen_scene, "mvp.pt"), map_location="cpu")

        scene_name = os.path.basename(base_scene)
        mesh_path = os.path.join(base_scene, scene_name + ".obj")
        uv_path = os.path.join(base_scene, "final_texture.png")
        mesh, uv_map = load_mesh_and_uv(mesh_path, uv_path, "cpu")
        mesh = vertex_transform(mesh, mesh_scale = self.mesh_scale)

        return mesh, uv_map[..., :3], gen_rgbs, mvps, int(timestep)

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx):
        if self.use_try_catch:
            try:
                return self.get_data(idx)
            except Exception as e:
                print(e)
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))
        else:
            return self.get_data(idx)

def collate_fn(batch):
    meshes = []
    uv_maps = []
    gen_rgbs = []
    mvps = []
    timesteps = []

    for mesh, uv_map, gen_rgb, mvp, timestep in batch:
        meshes.append(mesh)
        uv_maps.append(uv_map)
        gen_rgbs.append(gen_rgb)
        mvps.append(mvp)
        timesteps.append(timestep)
    
    gen_rgbs = torch.utils.data.default_collate(gen_rgbs)
    timesteps = torch.utils.data.default_collate(timesteps)

    return meshes, uv_maps, gen_rgbs, mvps, timesteps

def loader(train_batch_size, num_workers, **args):
    dataset = ObjaverseDataset(**args)
    return DataLoader(dataset, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

