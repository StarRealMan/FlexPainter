import os
import cv2
import numpy as np
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

def replace_nan(input_data, replace_value):
    if isinstance(input_data, torch.Tensor):
        return torch.where(torch.isnan(input_data), torch.tensor(replace_value, dtype=input_data.dtype, device=input_data.device), input_data)
    elif isinstance(input_data, list):
        return [
            torch.where(torch.isnan(t), torch.tensor(replace_value, dtype=t.dtype, device=t.device), t) if isinstance(t, torch.Tensor) else t
            for t in input_data
        ]
    else:
        raise TypeError("Input must be a torch.Tensor or a list of torch.Tensor")
