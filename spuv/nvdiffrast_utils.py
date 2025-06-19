import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


def rasterize_batched_geometry_maps(ctx, meshes, rasterize_height, rasterize_width):
    batch_size = len(meshes)
    position_maps = []
    normal_maps = []
    rasterization_masks = []
    for i in range(batch_size):
        mesh = meshes[i]
        position_map, normal_map, rasterization_mask = rasterize_geometry_maps(ctx, mesh, rasterize_height, rasterize_width)
        position_maps.append(position_map)
        normal_maps.append(normal_map)
        rasterization_masks.append(rasterization_mask)

    position_maps = torch.cat(position_maps, dim=0)

    rasterization_masks = torch.cat(rasterization_masks, dim=0)
    normal_maps = torch.cat(normal_maps, dim=0)
    return position_maps, normal_maps, rasterization_masks

def rasterize_geometry_maps(ctx, mesh, rasterize_height, rasterize_width):
    device = ctx.device
    mesh_v = mesh['v_pos'].to(device)
    mesh_f = mesh['t_pos_idx'].to(device)
    uvs_tensor = mesh['_v_tex'].to(device)
    indices_tensor = mesh['_t_tex_idx'].to(device)

    uv_clip = uvs_tensor[None, ...] * 2.0 - 1.0
    uv_clip_padded = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., :1]), torch.ones_like(uv_clip[..., :1])), dim=-1)
    rasterized_output, _ = ctx.rasterize(uv_clip_padded, indices_tensor.int(), (rasterize_height, rasterize_width))
    position_map, _ = ctx.interpolate_one(mesh_v, rasterized_output, mesh_f.int())

    mesh_normals = mesh['v_norm'].to(device)
    normal_map, _ = ctx.interpolate_one(mesh_normals, rasterized_output, mesh_f.int())
    normal_map = F.normalize(normal_map, dim=-1)

    rasterization_mask = rasterized_output[..., 3:4] > 0

    return position_map, normal_map, rasterization_mask


def render_batched_meshes(ctx, meshes, uv_map, mvp_mtx, image_height, image_width, background_color):
    batch_size = len(meshes)
    rgb = []
    with torch.cuda.amp.autocast(enabled=False):
        for i in range(batch_size):
            texture_map = uv_map[i].unsqueeze(0)
            render_rgb = render_rgb_from_texture_mesh(
                ctx, meshes[i], texture_map, mvp_mtx[i], image_height, image_width, background_color)
            rgb.append(render_rgb)

    rgb = torch.stack(rgb, dim=0)
    return rgb

def render_batched_light_meshes(ctx, light_ctx,
    FG_LUT, meshes, uv_map, mvp_mtx, camera_positions, image_height, image_width, background_color):
    batch_size = len(meshes)
    rgb = []
    for i in range(batch_size):
        texture_map = uv_map[i].unsqueeze(0)
        render_rgb = render_light_rgb_from_texture_mesh(
            ctx, light_ctx, FG_LUT, meshes[i], texture_map, mvp_mtx[i], camera_positions[i], image_height, image_width, background_color)
        rgb.append(render_rgb)

    rgb = torch.stack(rgb, dim=0)
    return rgb

def render_batched_xyz(ctx, meshes, mvp_mtx, image_height, image_width):
    batch_size = len(meshes)
    rgb = []
    for i in range(batch_size):
        render_rgb, mask = render_xyz_from_mesh(
            ctx, meshes[i], mvp_mtx[i], image_height, image_width)
        rgb.append(render_rgb)

    rgb = torch.stack(rgb, dim=0)
    return rgb

def render_xyz_from_mesh(ctx, mesh, mvp_matrix, image_height, image_width):
    device = mvp_matrix.device
    vertex_positions_clip = ctx.vertex_transform(mesh['v_pos'].to(device), mvp_matrix)
    rasterized_output, _ = ctx.rasterize(vertex_positions_clip, mesh['t_pos_idx'].to(device), (image_height, image_width))
    interpolated_positions, _ = ctx.interpolate_one(mesh['v_pos'].to(device), rasterized_output, mesh['t_pos_idx'].to(device))

    mask = rasterized_output[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rasterized_output, vertex_positions_clip, mesh['t_pos_idx'].to(device))

    batch_size = mvp_matrix.shape[0]
    rgb_foreground_batched = torch.zeros(batch_size, image_height, image_width, 3).to(interpolated_positions)
    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(interpolated_positions)

    selector = mask[..., 0]
    rgb_foreground_batched[selector] = interpolated_positions[selector]

    final_rgb = torch.lerp(rgb_background_batched, rgb_foreground_batched, mask_antialiased)
    final_rgb_aa = ctx.antialias(final_rgb, rasterized_output, vertex_positions_clip, mesh['t_pos_idx'].to(device))

    return final_rgb_aa, mask_antialiased

def render_pbr_view(
    light_ctx,
    FG_LUT,
    albedo,
    viewdirs,
    shading_normal,
    **kwargs,
):
    prefix_shape = albedo.shape[:-1]
    device = albedo.device

    metallic = 0.2 * torch.ones(*prefix_shape, 1, device=device, dtype=albedo.dtype)
    roughness = 0.5 * torch.ones(*prefix_shape, 1, device=device, dtype=albedo.dtype)

    v = -viewdirs
    n_dot_v = (shading_normal * v).sum(-1, keepdim=True)
    reflective = n_dot_v * shading_normal * 2 - v

    diffuse_albedo = (1 - metallic) * albedo

    fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)

    fg = dr.texture(
        FG_LUT.to(device),
        fg_uv.to(device).reshape(1, -1, 1, 2).contiguous(),
        filter_mode="linear",
        boundary_mode="clamp",
    ).reshape(*prefix_shape, 2)

    F0 = (1 - metallic) * 0.04 + metallic * albedo
    specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

    diffuse_light = light_ctx(shading_normal.to(device))
    specular_light = light_ctx(reflective.to(device), roughness.to(device))

    color = diffuse_albedo * diffuse_light + specular_albedo * specular_light
    color = color.clamp(0.0, 1.0)

    return color

def render_light_rgb_from_texture_mesh(
    ctx,
    light_ctx,
    FG_LUT,
    mesh,
    tex_map,
    mvp_matrix,
    camera_positions,
    image_height: int,
    image_width: int,
    background_color = torch.tensor([0.0, 0.0, 0.0]),
):

    vertex_positions_clip = ctx.vertex_transform(mesh['v_pos'], mvp_matrix)
    rast, _ = ctx.rasterize(vertex_positions_clip, mesh['t_pos_idx'], (image_height, image_width))
    mask = rast[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rast, vertex_positions_clip, mesh['t_pos_idx'])

    selector = mask[..., 0]

    gb_pos, _ = ctx.interpolate_one(mesh['v_pos'], rast, mesh['t_pos_idx'])
    gb_viewdirs = F.normalize(
        gb_pos - camera_positions[:, None, None, :], dim=-1
    )

    gb_normal, _ = ctx.interpolate_one(mesh['v_nrm'], rast, mesh['t_pos_idx'])
    gb_normal = F.normalize(gb_normal, dim=-1)

    extra_geo_info = {}
    extra_geo_info["shading_normal"] = gb_normal[selector]

    interpolated_texture_coords, _ = ctx.interpolate_one(mesh['_v_tex'], rast,
                                                              mesh['_t_tex_idx'])
    material_info = {
        "albedo": texture_map_to_rgb(tex_map, interpolated_texture_coords)[selector]
    }

    rgb_foreground = render_pbr_view(
        light_ctx,
        FG_LUT,
        viewdirs=gb_viewdirs[selector],
        **extra_geo_info,
        **material_info,
    )

    batch_size, height, width = mask.shape[:3]
    rgb_foreground_batched = torch.zeros(batch_size, image_height, image_height, 3).to(rgb_foreground)
    rgb_foreground_batched[selector] = rgb_foreground

    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched += background_color.view(1, 1, 1, 3).to(rgb_foreground)
    # Use the anti-aliased mask for blending
    final_rgb = torch.lerp(rgb_background_batched, rgb_foreground_batched, mask_antialiased)
    final_rgb_aa = ctx.antialias(final_rgb, rast, vertex_positions_clip, mesh['t_pos_idx'])

    return final_rgb_aa

def render_rgb_from_texture_mesh(
    ctx,
    mesh,
    tex_map,
    mvp_matrix,
    image_height: int,
    image_width: int,
    background_color = torch.tensor([0.0, 0.0, 0.0]),
):
    batch_size = mvp_matrix.shape[0]
    tex_map = tex_map.unsqueeze(0).contiguous()

    vertex_positions_clip = ctx.vertex_transform(mesh['v_pos'], mvp_matrix)
    rasterized_output, _ = ctx.rasterize(vertex_positions_clip, mesh['t_pos_idx'], (image_height, image_width))
    mask = rasterized_output[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rasterized_output, vertex_positions_clip, mesh['t_pos_idx'])

    interpolated_texture_coords, _ = ctx.interpolate_one(mesh['_v_tex'], rasterized_output, mesh['_t_tex_idx'])
    rgb_foreground = texture_map_to_rgb(tex_map.float(), interpolated_texture_coords)
    rgb_foreground_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched += background_color.view(1, 1, 1, 3).to(rgb_foreground)

    selector = mask[..., 0]
    rgb_foreground_batched[selector] = rgb_foreground[selector]

    # Use the anti-aliased mask for blending
    final_rgb = torch.lerp(rgb_background_batched, rgb_foreground_batched, mask_antialiased)
    final_rgb_aa = ctx.antialias(final_rgb, rasterized_output, vertex_positions_clip, mesh['t_pos_idx'])

    return final_rgb_aa

def render_rgb_mask_from_texture_mesh(
    ctx,
    mesh,
    tex_map,
    mvp_matrix,
    image_height: int,
    image_width: int,
    background_color = torch.tensor([0.0, 0.0, 0.0]),
):
    batch_size = mvp_matrix.shape[0]
    tex_map = tex_map.unsqueeze(0).contiguous()

    vertex_positions_clip = ctx.vertex_transform(mesh['v_pos'], mvp_matrix)
    rasterized_output, _ = ctx.rasterize(vertex_positions_clip, mesh['t_pos_idx'], (image_height, image_width))
    mask = rasterized_output[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rasterized_output, vertex_positions_clip, mesh['t_pos_idx'])

    interpolated_texture_coords, _ = ctx.interpolate_one(mesh['_v_tex'], rasterized_output, mesh['_t_tex_idx'])
    rgb_foreground = texture_map_to_rgb(tex_map.float(), interpolated_texture_coords)
    rgb_foreground_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched += background_color.view(1, 1, 1, 3).to(rgb_foreground)

    selector = mask[..., 0]
    rgb_foreground_batched[selector] = rgb_foreground[selector]

    # Use the anti-aliased mask for blending
    final_rgb = torch.lerp(rgb_background_batched, rgb_foreground_batched, mask_antialiased)
    final_rgb_aa = ctx.antialias(final_rgb, rasterized_output, vertex_positions_clip, mesh['t_pos_idx'])

    return torch.cat([final_rgb_aa, mask_antialiased.float()], dim=-1)

def texture_map_to_rgb(tex_map, uv_coordinates):
    return dr.texture(tex_map.float(), uv_coordinates)

def render_normal_from_mesh(ctx, mesh, mvp_matrix, image_height, image_width):
    device = mvp_matrix.device
    vertices = mesh['v_pos'].to(device)
    normals = mesh['v_norm'].to(device)
    faces = mesh['t_pos_idx'].to(device)

    vertex_positions_clip = ctx.vertex_transform(vertices, mvp_matrix)
    rast, _ = ctx.rasterize(vertex_positions_clip, faces.int(), (image_height, image_width))
    interpolated_normals, _ = ctx.interpolate_one(normals, rast, faces)
    interpolated_normals = F.normalize(interpolated_normals, dim=-1)

    return interpolated_normals
