import os

import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange

from spuv.nvdiffrast_utils import *


def bake_render_to_uv_single_concat_disable_occlusion(
    ctx,
    mesh,
    mvp_matrix,
    views,
    views_xyz,
    view_rast,
    pixel_3d_positions  # position map
):
    data_type = pixel_3d_positions.dtype
    device = pixel_3d_positions.device
    num_views, height, width, feature_channel = views.shape
    H, W, C = pixel_3d_positions.shape

    all_uv_colors = torch.zeros((num_views, H, W, feature_channel), dtype=data_type, device=device)
    uv_colors = torch.zeros((num_views, H, W, feature_channel), dtype=data_type, device=device)
    uv_color_masks = torch.zeros((num_views, H, W), dtype=torch.bool, device=device)
    uv_color_weights = torch.zeros((num_views, H, W), dtype=data_type, device=device)

    uv_triangle_indices = get_uv_rast(ctx, mesh, device, texture_size=H)
    num_triangles = mesh['t_pos_idx'].shape[0]

    for view_index in range(num_views):
        visible_triangle_indices = torch.unique(view_rast[view_index, :, :, 3]).long()
        visible_triangles = torch.zeros((num_triangles + 1,), dtype=torch.bool, device=device)
        visible_triangles[visible_triangle_indices] = True
        visible_triangles[0] = False

        pixel_visible_mask, pixel_2d_positions_ndc = compute_visibility_mask_with_occlusion(
            ctx,
            pixel_3d_positions, views_xyz[view_index], visible_triangles, mvp_matrix[view_index: view_index + 1],
            uv_triangle_indices
        )
        pixel_colors = compute_visible_rgb(views[view_index], pixel_2d_positions_ndc)

        all_pixel_3d_positions = pixel_3d_positions.view(-1, 3).float()
        all_pixel_2d_positions = ctx.vertex_transform(all_pixel_3d_positions, mvp_matrix)
        all_pixel_2d_positions_ndc = all_pixel_2d_positions[:, :, :3] / all_pixel_2d_positions[:, :, 3:4]
        all_pixel_colors = compute_visible_rgb(views[view_index], all_pixel_2d_positions_ndc)

        all_uv_colors[view_index] = all_pixel_colors.view(H, W, feature_channel)
        uv_colors[view_index][pixel_visible_mask] = pixel_colors
        uv_color_masks[view_index] = pixel_visible_mask
        uv_color_weights[view_index] = pixel_visible_mask.float()

    uv_new_color = rearrange(uv_colors, 'v h w c -> (v c) h w')
    #uv_color_weights = rearrange(uv_color_weights, 'v h w -> v h w')
    return uv_new_color, uv_color_weights

def bake_image_feature_to_uv(ctx, mesh, image_info, position_map):
    """
    :param mesh: List of mesh (len=B)
    :param image_info: Dict of image info
    :param position_map: B H W 3
    :param device:
    :return:
    """

    position_map = position_map
    # B Nv 4 4
    mvp_matrix = image_info['mvp_mtx']
    # B Nv H W 3
    views = image_info['rgb']
    batch_size, num_views, height, width, channels = views.shape
    device = views.device
    rast = get_view_rast(ctx, mesh, mvp_matrix, height, width, device)

    baked_colors = []
    baked_masks = []
    for b in range(batch_size):
        views_xyz_b, mask = render_xyz_from_mesh(ctx, mesh[b], mvp_matrix[b], height, width)

        uv_new_color, uv_color_mask = bake_render_to_uv_single_concat_disable_occlusion(
            ctx,
            mesh[b],
            mvp_matrix[b],
            views[b],
            views_xyz_b,
            rast[b],
            position_map[b],
        )

        baked_colors.append(uv_new_color)
        baked_masks.append(uv_color_mask)

    return torch.stack(baked_colors, dim=0), torch.stack(baked_masks, dim=0)

def compute_visibility_mask_with_occlusion(ctx, pixel_3d_positions, view_xyz, visible_triangles_mask, mvp_matrix,
                                           uv_triangle_indices):
    H, W, _ = pixel_3d_positions.shape

    # Apply visible triangles mask to get visible 3D positions and flatten it.
    pixel_visible_mask_flat = visible_triangles_mask[uv_triangle_indices].view(-1)
    pixel_3d_positions_visible = pixel_3d_positions.reshape(-1, 3).contiguous()[pixel_visible_mask_flat].float()

    # Transform 3D positions to 2D using the provided transformation matrix.
    pixel_2d_positions = ctx.vertex_transform(pixel_3d_positions_visible, mvp_matrix)
    pixel_2d_positions_ndc = pixel_2d_positions[:, :, :3] / pixel_2d_positions[:, :, 3:4]

    pixel_interpolate_axis = pixel_2d_positions_ndc[0, :, :2].unsqueeze(0).unsqueeze(0)
    pixel_xyz = F.grid_sample(
            view_xyz.unsqueeze(0).permute(0, 3, 1, 2),  # Change view to 'batch x channel x height x width'
            pixel_interpolate_axis,  # Change positions to 'batch x 1 x N x 2'
            mode="bilinear", align_corners=True
        ).squeeze(0).squeeze(1).permute(1, 0)  # Reshape back to 'N x channel'

    # Compute occlusion mask based on the proximity in 3D space.
    occlusion_threshold = 0.01  # TODO: Hard Coding; Adjustable threshold for occlusion.

    occlusion_mask = torch.norm(pixel_xyz - pixel_3d_positions_visible, dim=-1) < occlusion_threshold

    # Combine occlusion information with the visibility mask.
    pixel_visible_mask_flat[pixel_visible_mask_flat.nonzero().squeeze(1)] &= occlusion_mask

    # Reshape the mask back to its original dimensions.
    updated_pixel_visible_mask = pixel_visible_mask_flat.view(H, W)

    # Update pixel_2d_positions_ndc based on the updated visibility mask.
    updated_pixel_3d_positions_visible = pixel_3d_positions.view(-1, 3)[pixel_visible_mask_flat].float()
    updated_pixel_2d_positions = ctx.vertex_transform(updated_pixel_3d_positions_visible, mvp_matrix)
    updated_pixel_2d_positions_ndc = updated_pixel_2d_positions[:, :, :3] / updated_pixel_2d_positions[:, :, 3:4]

    return updated_pixel_visible_mask, updated_pixel_2d_positions_ndc

def compute_visible_rgb(view, pixel_2d_positions_ndc):
    # Sample the RGB and XYZ values from the view using bilinear interpolation.
    pixel_interpolate_axis = pixel_2d_positions_ndc[0, :, :2].unsqueeze(0).unsqueeze(0)

    data_type = view.dtype
    pixel_rgb = F.grid_sample(
        view.unsqueeze(0).permute(0, 3, 1, 2).float(),  # Change view to 'batch x channel x height x width'
        pixel_interpolate_axis.float(),  # Change positions to 'batch x 1 x N x 2'
        mode="bilinear", align_corners=True
    ).squeeze(0).squeeze(1).permute(1, 0)  # Reshape back to 'N x channel'

    return pixel_rgb.to(dtype=data_type)

def get_view_rast(ctx, mesh, mvp_mtx, view_h, view_w, device):
    B = len(mesh)
    view_rast = []
    for b in range(B):
        v_pos = mesh[b]['v_pos'].to(device)
        t_pos_idx = mesh[b]['t_pos_idx'].to(device)
        v_pos_clip = ctx.vertex_transform(v_pos, mvp_mtx[b])
        rast, _ = ctx.rasterize(v_pos_clip,
                                t_pos_idx,
                                (view_h, view_w))
        view_rast.append(rast)
    return torch.stack(view_rast, dim=0)

def get_uv_rast(ctx, mesh, device, texture_size=1024):
    uv = mesh['_v_tex'].to(device)
    t_tex_index = mesh['_t_tex_idx'].to(device)

    uv_clip = uv * 2.0 - 1.0
    uv_clip4 = torch.cat(
        (
            uv_clip,
            torch.zeros_like(uv_clip[..., 0:1]),
            torch.ones_like(uv_clip[..., 0:1]),
        ),
        dim=-1,
    )

    rast_uv, _ = ctx.rasterize_one(
        uv_clip4,
        # mesh['_t_tex_idx'][0].cuda(),
        t_tex_index,
        (texture_size, texture_size)
    )
    uv_tri_idx = rast_uv[..., 3].long()

    return uv_tri_idx
