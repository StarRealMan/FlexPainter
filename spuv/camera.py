import math

import numpy as np
import torch
import torch.nn.functional as F


def get_c2w(
        azimuth_deg,
        elevation_deg,
        camera_distances,):
    assert len(azimuth_deg) == len(elevation_deg) == len(camera_distances)
    n_views = len(azimuth_deg)
    device = azimuth_deg.device
    #camera_distances = torch.full_like(elevation_deg, dis)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w

def get_c2w_from_uniform_azimuth(ele, dis, n_views):
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, ele)
    camera_distances = torch.full_like(elevation_deg, dis)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w
