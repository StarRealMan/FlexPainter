import torch

from spuv.nvdiffrast_utils import render_rgb_from_texture_mesh, render_normal_from_mesh, rasterize_geometry_maps
from model.utils.feature_baking import bake_image_feature_to_uv

def normalize_depth(depth, mask):
    depth = depth * mask + 100 * (1-mask)
    inv_depth = 1 / depth
    inv_depth_min = inv_depth * mask + 100 * (1-mask)
    
    max_ = torch.max(inv_depth)
    min_ = torch.min(inv_depth_min)
    inv_depth = (inv_depth - min_) / (max_ - min_)
    inv_depth = inv_depth.clamp(0,1)

    return inv_depth

def position_to_depth(position_ndc, c2w_matrices):
    V, H, W, _ = position_ndc.shape
    device = position_ndc.device

    ones = torch.ones((V, H, W, 1), device=device)
    position_ndc_homo = torch.cat((position_ndc, ones), dim=-1)
    position_ndc_homo = position_ndc_homo.view(V, -1, 4)

    inv_c2w_matrices = torch.linalg.inv(c2w_matrices)
    position_view_homo = torch.bmm(inv_c2w_matrices, position_ndc_homo.transpose(1, 2))
    position_view_homo = position_view_homo.transpose(1, 2)

    z_view = -position_view_homo[..., 2]
    depth_view = z_view.view(V, H, W, 1)

    return depth_view

def generate_ray_image(mvp_matrix, width, height):
    view_num = mvp_matrix.shape[0]
    device = mvp_matrix.device

    y, x = torch.meshgrid(
        torch.linspace(0, height - 1, height),
        torch.linspace(0, width - 1, width)
    )
    x = x.to(device)
    y = y.to(device)
    ndc_x = (x / (width - 1)) * 2 - 1
    ndc_y = (y / (height - 1)) * 2 - 1
    ndc_z = torch.ones_like(ndc_x)

    pixel_homogeneous = torch.stack([ndc_x, ndc_y, ndc_z, torch.ones_like(ndc_x)], dim=-1)
    pixel_homogeneous = pixel_homogeneous.unsqueeze(0).repeat(view_num, 1, 1, 1)

    inv_mvp = torch.linalg.inv(mvp_matrix)
    world_coords = torch.einsum('vij,vhwj->vhwi', inv_mvp, pixel_homogeneous)

    rays = world_coords[..., :3]
    rays = rays / torch.norm(rays, dim=-1, keepdim=True)

    return rays

def rotate_c2w(rays):
    V, H, W, _ = rays.shape
    device = rays.device

    Rc2w = torch.tensor([
        [1, 0, 0], 
        [0, 0, 1], 
        [0, -1, 0],   
    ], device=device, dtype=rays.dtype)
    rays = rays.reshape(-1, 3)
    rays = torch.matmul(Rc2w, rays.transpose(0, 1)).transpose(0, 1)
    rays = rays.view(V, H, W, 3)

    return rays

def data_process(ctx, meshes, uv_maps, mvps, gen_rgbs, bg):
    tex_h, tex_w, _ =  uv_maps[0].shape
    _, img_h, img_w, _ = gen_rgbs[0].shape

    uv_positions = []
    uv_normals = []
    uv_masks = []
    uv_bakess = []
    uv_bake_maskss = []
    rgbs = []
    for mesh, uv_map, mvp, gen_rgb in zip(meshes, uv_maps, mvps, gen_rgbs):
        rgb = render_rgb_from_texture_mesh(ctx, mesh, uv_map[..., :3], mvp, img_h, img_w, bg)
        normal = render_normal_from_mesh(ctx, mesh, mvp, img_h, img_w)
        rays_d = generate_ray_image(mvp, img_h, img_w)
        rays_d = rotate_c2w(rays_d)
        score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
        score = torch.abs(score)
        feature = torch.cat([rgb, gen_rgb, rays_d, score], dim=-1)

        uv_position, uv_normal, uv_mask = rasterize_geometry_maps(ctx, mesh, tex_h, tex_w)
        image_info = {
            "mvp_mtx": mvp.unsqueeze(0),
            "rgb": feature.unsqueeze(0)
        }

        uv_bakes, uv_bake_masks = bake_image_feature_to_uv(
            ctx, [mesh], image_info, uv_position
        )
        uv_bakes = uv_bakes.view(-1, feature.shape[-1], tex_h, tex_w)
        uv_bake_masks = uv_bake_masks.view(-1, 1, tex_h, tex_w)

        uv_positions.append(uv_position)
        uv_normals.append(uv_normal)
        uv_masks.append(uv_mask)
        uv_bakess.append(uv_bakes)
        uv_bake_maskss.append(uv_bake_masks)
        rgbs.append(rgb)
    
    uv_positions = torch.cat(uv_positions, dim=0).permute(0, 3, 1, 2)
    uv_normals = torch.cat(uv_normals, dim=0).permute(0, 3, 1, 2)
    uv_masks = torch.cat(uv_masks, dim=0).permute(0, 3, 1, 2)
    uv_bakess = torch.stack(uv_bakess, dim=0)
    uv_bake_maskss = torch.stack(uv_bake_maskss, dim=0)
    rgbs = torch.stack(rgbs, dim=0)
        
    return uv_positions, uv_normals, uv_masks, uv_bakess, uv_bake_maskss, rgbs
