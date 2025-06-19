import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def downsample_mask(mask_map: torch.Tensor) -> torch.Tensor:
    """
    Downsample the mask map by a factor of 2.
    :param mask_map: Tensor of shape (B, 1, H, W).
    :return: Downsampled mask map.
    """
    return F.max_pool2d(-1 * mask_map.float(), kernel_size=2, stride=2, padding=0) < -0.5

def downsample_chart(chart_map: torch.Tensor) -> torch.Tensor:
    """
    Custom downsampling of the chart map.
    :param chart_map: Tensor of shape (B, 1, H, W) with integer values.
    :return: Downsampled chart map.
    """
    B, _, H, W = chart_map.shape
    chart_map_reshaped = chart_map.view(B, H//2, 2, W//2, 2)
    window_equal = torch.eq(chart_map_reshaped, chart_map_reshaped[:, :, 0:1, :, 0:1]).all(dim=2).all(dim=3)
    return torch.where(window_equal, chart_map_reshaped[:, :, 0, :, 0], torch.tensor(0, device=chart_map.device))

def upsample_mask(mask_map: torch.Tensor) -> torch.Tensor:
    """
    Upsample the mask map by a factor of 2 using nearest neighbor interpolation.
    :param mask_map: Tensor of shape (B, 1, H, W).
    :return: Upsampled mask map.
    """
    return F.interpolate(mask_map.float(), scale_factor=2, mode='nearest') > 0.5

def upsample_chart(chart_map: torch.Tensor) -> torch.Tensor:
    """
    Custom upsampling of the chart map using nearest neighbor interpolation.
    :param chart_map: Tensor of shape (B, 1, H, W) with integer values.
    :return: Upsampled chart map.
    """
    return F.interpolate(chart_map.float(), scale_factor=2, mode='nearest').int()

def downsample_feature_with_mask(feature_map, mask_map):
    """
    Downsample the feature map by a factor of 2, applying the mask.
    This function works with both a single tensor and a list of tensors.
    :param feature_map: Tensor or list of Tensors, each of shape (B, C, H, W).
    :param mask_map: Tensor of shape (B, 1, H, W).
    :return: Downsampled feature map, as a single tensor or list of tensors.
    """

    data_type = feature_map[0].dtype
    mask_map_downsampled = downsample_mask(mask_map).to(dtype=data_type)
    def downsample_single_feature_map(feature_map):
        feature_map_downsampled = F.avg_pool2d(feature_map, kernel_size=2, stride=2, padding=0)
        feature_map_downsampled *= mask_map_downsampled.type_as(feature_map)
        return feature_map_downsampled

    # Check if feature_map is a list and apply processing to each element
    if isinstance(feature_map, list):
        return [downsample_single_feature_map(fm) for fm in feature_map], mask_map_downsampled
    else:
        # Apply processing to a single tensor
        return downsample_single_feature_map(feature_map), mask_map_downsampled

def upsample_feature_with_mask(feature_map, mask_map):
    """
    Upsample the feature map by a factor of 2, applying the mask.
    This function works with both a single tensor and a list of tensors.
    :param feature_map: Tensor or list of Tensors, each of shape (B, C, H, W).
    :param mask_map: Tensor of shape (B, 1, H, W).
    :return: Upsampled feature map, as a single tensor or list of tensors.
    """

    mask_map_upsampled = upsample_mask(mask_map).float()
    def upsample_single_feature_map(feature_map):
        feature_map_upsampled = F.interpolate(feature_map, scale_factor=2, mode='bilinear', align_corners=True)
        feature_map_upsampled *= mask_map_upsampled.type_as(feature_map)
        return feature_map_upsampled

    # Check if feature_map is a list and apply processing to each element
    if isinstance(feature_map, list):
        return [upsample_single_feature_map(fm) for fm in feature_map], mask_map_upsampled
    else:
        # Apply processing to a single tensor
        return upsample_single_feature_map(feature_map), mask_map_upsampled

def point2uv(point_feature, mask_map):
    """
    :param point_feature: Tensor, shape = (B, N, C_out)
    :param mask_map: Tensor, shape = (B, 1, H, W)
    :return: Tensor, shape = (B, C_out, H, W)
    """
    output_point_feature_map = rearrange(point_feature,
                                         "B (H W) C -> B C H W",
                                         H=mask_map.shape[2],
                                         W=mask_map.shape[3],
                                         )
    return output_point_feature_map


def uv2point(feature_map, mask_map, position_map):
    """
    :param feature_map: Tensor, shape = (B, C_in, H, W)
    :param mask_map: Tensor, shape = (B, 1, H, W)
    :param position_map: Tensor, shape = (B, 3, H, W)
    :return: Tensor, shape = (B, N, C_in), Tensor, shape = (B, N), Tensor, shape = (B, N, 3)
    """
    point_feature = rearrange(feature_map, "B C H W -> B C (H W)")
    point_feature = rearrange(point_feature, "B C N -> B N C")

    point_mask = rearrange(mask_map, "B 1 H W -> B H W")
    point_mask = rearrange(point_mask, "B H W -> B (H W)")

    point_position = rearrange(position_map, "B C H W -> B C (H W)")
    point_position = rearrange(point_position, "B C N -> B N C")

    return point_feature, point_mask, point_position
