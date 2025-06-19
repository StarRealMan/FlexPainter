import os
import torch
from torchvision.utils import save_image

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    else:
        return data

def save_res(uv_maps, uv_position, uv_normal, uv_mask, uv_bakes, uv_bake_masks, output_dir):
    save_image(uv_maps[0].permute(2, 0, 1), os.path.join(output_dir, "gt.png"))
    save_image((uv_position+1)/2, os.path.join(output_dir, "position.png"))
    save_image((uv_normal+1)/2, os.path.join(output_dir, "normal.png"))
    save_image(uv_mask.float(), os.path.join(output_dir, "mask.png"))
    save_image(uv_bakes.squeeze(0)[:, :3], os.path.join(output_dir, "bakes.png"))
    save_image((uv_bakes.squeeze(0)[:, 3:6]+1)/2, os.path.join(output_dir, "rays.png"))
    save_image(uv_bakes.squeeze(0)[:, 6:7], os.path.join(output_dir, "scores.png"))
    save_image(uv_bake_masks.squeeze(0), os.path.join(output_dir, "bake_masks.png"))

    occ_mask = uv_mask
    for i in range(uv_bake_masks.shape[1]):
        occ_mask = occ_mask * uv_bake_masks[:, i]
    save_image(occ_mask.float(), os.path.join(output_dir, f"occ_mask.png"))
