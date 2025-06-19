from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from torchsparse import nn as spnn
from torchsparse import SparseTensor

from .utils.uv_operators import *
from .utils.emb_utils import *
from .utils.sparse_utils import *


from spuv.rasterize import NVDiffRasterizerContext
from model.misc import get_device
from model.base import BaseModule

from model.ptv3_model_texgen import Point as PTV3_Point
from model.ptv3_model_texgen import (
    PointSequential,
    TimeBlock,
)

class OutpainterNet(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 3
        out_channels: int = 3
        num_layers: Tuple[int] = (1, 1, 2, 4)
        block_out_channels: Tuple[int] = (32, 64, 128, 256)
        dropout: Tuple[float] = (0.0, 0.0, 0.0, 0.0)
        voxel_size: Tuple[float] = (0.1, 0.1, 0.1, 0.1)
        block_type: Tuple[str] = ("point_uv", "point_uv", "point_uv", "point_uv")
        window_size: Tuple[int] = (32, 32, 32, 32)
        skip_input: bool = True
        skip_type: str = "baked_texture"
        num_heads: Tuple[int] = (4, 4, 4, 4)
        point_block_num: Tuple[int] = (2, 2, 2, 2)
        use_uv_head: bool = True
        weights: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_channels = self.cfg.in_channels
        out_channels = self.cfg.out_channels
        num_layers = self.cfg.num_layers
        block_out_channels = self.cfg.block_out_channels
        voxel_size = self.cfg.voxel_size
        block_type = self.cfg.block_type
        dropout = self.cfg.dropout
        window_size = self.cfg.window_size
        num_heads = self.cfg.num_heads
        point_block_num = self.cfg.point_block_num
        use_uv_head = self.cfg.use_uv_head

        device = get_device()
        self.ctx = NVDiffRasterizerContext('cuda', device)

        self.block_out_channels = block_out_channels

        self.input_conv = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=1, stride=1, padding=0)

        for scale in range(len(block_out_channels)):
            setattr(self, f"down{scale}",
                    PointUVStage(
                        block_out_channels[scale],
                        num_layers[scale],
                        dropout=dropout[scale],
                        voxel_size=voxel_size[scale],
                        conv_type=block_type[scale],
                        window_size=window_size[scale],
                        num_heads=num_heads[scale],
                        point_block_num=point_block_num[scale],
                        use_uv_head=use_uv_head,
                    ))

            if scale < len(block_out_channels) - 1:
                setattr(self, f"post_conv_down{scale}", nn.Conv2d(
                    block_out_channels[scale], block_out_channels[scale + 1], kernel_size=1, stride=1, padding=0))

        for scale in reversed(range(len(block_out_channels) - 1)):
            if scale < len(block_out_channels) - 1:
                setattr(self, f"pre_conv_up{scale}", nn.Conv2d(
                    block_out_channels[scale + 1],
                    block_out_channels[scale],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False))

                setattr(self, f"skip_conv{scale}",
                        nn.Conv2d(2 * block_out_channels[scale],
                                  block_out_channels[scale],
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False))

                setattr(self, f"skip_layer_norm{scale}",
                        nn.LayerNorm(block_out_channels[scale], elementwise_affine=True))

            setattr(self, f"up{scale}",
                    PointUVStage(
                        block_out_channels[scale],
                        num_layers[scale],
                        dropout=dropout[scale],
                        voxel_size=voxel_size[scale],
                        conv_type=block_type[scale],
                        window_size=window_size[scale],
                        num_heads=num_heads[scale],
                        point_block_num=point_block_num[scale],
                        use_uv_head=use_uv_head,
                    ))

            in_channels = block_out_channels[scale]

        self.condition_embedder = ConditionEmbedding(clip_condition=True, double_condition=True)
        if self.cfg.skip_input and self.cfg.skip_type == "adaptive":
            self.ada_skip_scale = nn.Linear(1024, 2, bias=True)
            self.ada_skip_map = nn.Conv2d(in_channels+10, 2 * out_channels, kernel_size=1, stride=1, padding=0)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.weight_initialization()

    def weight_initialization(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for name, module in self.named_modules():
            # Zero-out adaLN modulation layers in DiT blocks:
            if "adaLN_modulation.linear" in name:
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)
            if "timestep_embedder.linear" in name or "clip_embedding_projection.linear" in name:
                nn.init.normal_(module.weight, std=0.02)
            # do not zero init these two adaptive layers at the same time, causing zero gradient!
            if "ada_skip_scale" in name:
                nn.init.normal_(module.weight, std=0.02)
            if "ada_skip_map" in name:
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 0)
            if "clip_embedding_projection2.linear2" in name:
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 0)
            if "global_context_embedder.linear2" in name:
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)
            if "abs_pos_embed.fc2" in name:
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)
            if "dit_blocks.q" in name or "dit_blocks.k" in name or "dit_blocks.k" in name or "dit_blocks.proj" in name:
                nn.init.normal_(module.weight, std=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if "point_gate.1" in name:
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)

    def forward(self,
                x_dense,
                mask_map,
                position_map,
                timestep,
                clip_embeddings,
                mesh,
                baked_texture,
                baked_weights,
                condition_drop,
                ):
        """
        :param x_dense: dense feature map
        :param mask_map: dense mask map
        :param position_map: dense position map
        :return:
        """
        skip_x = x_dense

        # for cfg
        input_embeddings = []
        condition_drop = condition_drop.unsqueeze(-1)
        for i, _ in enumerate(clip_embeddings):
            clip_embedding_null = torch.zeros_like(clip_embeddings[i], device=x_dense.device, dtype=x_dense.dtype)
            clip_embedding = condition_drop * clip_embedding_null + (1 - condition_drop) * clip_embeddings[i]
            input_embeddings.append(clip_embedding)

        baked_texture = baked_texture * 2 - 1

        x_concat = torch.cat([x_dense, position_map, baked_texture, baked_weights], dim=1)
        x_dense = self.input_conv(x_concat) * mask_map
        if torch.isnan(x_dense).any():
            print("x_dense has NaN values")
            breakpoint()

        pyramid_features = []
        pyramid_mask = []
        pyramid_position = []

        condition_embedding = self.condition_embedder(timestep, input_embeddings)

        for scale in range(len(self.block_out_channels)):
            x_dense = getattr(self, f"down{scale}")(
                x_dense,
                mask_map,
                position_map,
                condition_embedding,
                ctx=self.ctx,
                mesh=mesh,
                feature_info=None,
            )

            if scale < len(self.block_out_channels) - 1:
                pyramid_features.append(x_dense)
                pyramid_mask.append(mask_map)
                pyramid_position.append(position_map)

                feature_list, mask_map = downsample_feature_with_mask([x_dense, position_map], mask_map)
                x_dense, position_map = feature_list

                x_dense = getattr(self, f"post_conv_down{scale}")(x_dense)

        for scale in reversed(range(len(self.block_out_channels) - 1)):
            if scale < len(self.block_out_channels) - 1:
                x_dense = getattr(self, f"pre_conv_up{scale}")(x_dense)

                x_dense, _ = upsample_feature_with_mask(x_dense, mask_map)
                mask_map = pyramid_mask[scale]
                position_map = pyramid_position[scale]

                x_dense = torch.cat([x_dense, pyramid_features[scale]], dim=1)
                x_dense = getattr(self, f"skip_conv{scale}")(x_dense)

                B, C, H, W = x_dense.shape
                x_dense = rearrange(x_dense, "B C H W -> (B H W) C")
                x_dense = getattr(self, f"skip_layer_norm{scale}")(x_dense)
                x_dense = rearrange(x_dense, "(B H W) C -> B C H W", B=B, H=H)

            x_dense = getattr(self, f"up{scale}")(
                x_dense,
                mask_map,
                position_map,
                condition_embedding,
                ctx=self.ctx,
                mesh=mesh,
                feature_info=None,
            )

        x_output = self.output_conv(x_dense)

        addition_info = {
            "pyramid_features": pyramid_features,
            "pyramid_mask": pyramid_mask,
            "pyramid_position": pyramid_position,
            "baked_texture": baked_texture,
            "baked_weights": baked_weights,
        }

        if self.cfg.skip_input:
            if self.cfg.skip_type == "baked_texture":
                return x_output + baked_weights * baked_texture, addition_info
            elif self.cfg.skip_type == "noise_input":
                return x_output + skip_x, addition_info
            elif self.cfg.skip_type == "adaptive":
                skip_scale = self.ada_skip_scale(condition_embedding)
                x0_scale, input_scale = skip_scale.chunk(2, dim=1)
                x0_scale = x0_scale.unsqueeze(-1).unsqueeze(-1)
                input_scale = input_scale.unsqueeze(-1).unsqueeze(-1)
                skip_map = self.ada_skip_map(torch.cat([x_concat, x_dense], dim=1))
                output_scale_map, skip_scale_map = skip_map.chunk(2, dim=1)

                x1 = (1-output_scale_map) * x_output
                x2 = skip_scale_map * (x0_scale * baked_texture + input_scale * skip_x)
                x_output = x1 + x2

                addition_info["skip_scale_map"] = skip_scale_map
                addition_info["output_scale_map"] = output_scale_map
                addition_info["skip_scale_rgb"] = x2
                addition_info["output_scale_input"] = x1

                return x_output, addition_info

        else:
            return x_output, addition_info

class PointUVStage(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers,
        dropout,
        window_size,
        use_uv_head,
        point_block_num=2,
        num_heads=4,
        voxel_size=0.01,
        conv_type="point_uv",
        **kwargs
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        order_list = ["z", "z-trans", "hilbert", "hilbert-trans"]
        if conv_type != "uv_dit":
            for i in range(num_layers):
                order = order_list[i % len(order_list)]
                block = ResidualSparsePointUVBlock(in_channels, voxel_size, conv_type, dropout, order, **kwargs)
                self.layers.append(block)
        elif conv_type == "uv_dit":
            block = UVPTVAttnStage(in_channels,
                                   voxel_size,
                                   num_layers,
                                   num_heads,
                                   point_block_num,
                                   order_list,
                                   window_size,
                                   use_uv_head,
                                   **kwargs)
            self.layers.append(block)
        else:
            raise ValueError("Invalid conv_type")

    def forward(self, x, mask_map, position_map=None, condition_embedding=None, ctx=None,
                mesh=None,
                feature_info=None,):
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                mask_map,
                position_map,
                condition_embedding,
                ctx=ctx,
                mesh=mesh,
                feature_info=feature_info,
            )

        return x

class ConditionEmbedding(nn.Module):
    def __init__(self, clip_condition=True, double_condition=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.clip_condition = clip_condition
        if clip_condition:
            if double_condition:
                self.emb = CombinedTimestepClipEmbeddings(clip_embedding_dim=[1024, 768],
                                                          embedding_dim=1024)
            else:
                self.emb = CombinedTimestepClipEmbeddings(clip_embedding_dim=1024,
                                                          embedding_dim=1024)
        else:
            self.emb = TimestepEmbeddings(embedding_dim=1024)

    def forward(self, timestep, clip_embedding):
        if self.clip_condition and clip_embedding is not None:
            condition_embedding = self.emb(timestep, clip_embedding, hidden_dtype=clip_embedding[0].dtype)
        else:
            condition_embedding = self.emb(timestep, hidden_dtype=clip_embedding.dtype)

        return condition_embedding

# The Core Block that has a hybrid 2D-3D structure
class ResidualSparsePointUVBlock(nn.Module):
    def __init__(self, in_channels, voxel_size, conv_type, dropout, order, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.conv_type = conv_type
        if conv_type == "uv":
            self.norm1 = PixelNorm(in_channels, layer_norm=True, affine=True)
            self.norm2 = PixelNorm(in_channels, layer_norm=True, affine=False)
            self.adaLN_modulation = UVAdaLayerNormZero(in_channels, in_channels)
            self.conv_layer1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv_layer2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        elif conv_type == "point_uv":
            self.norm1 = PixelNorm(in_channels, layer_norm=True, affine=True)
            self.norm2 = PixelNorm(in_channels, layer_norm=True, affine=False)
            self.adaLN_modulation = UVAdaLayerNormZero(in_channels, in_channels)
            self.point_adaLN_modulation = UVAdaLayerNormZero(in_channels, in_channels)
            self.conv_layer1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv_layer2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            self.point_block = PointBlock(in_channels)
        elif conv_type == "point":
            self.point_adaLN_modulation = UVAdaLayerNormZero(in_channels, in_channels)
            self.point_block = PointBlock(in_channels)
        else:
            raise ValueError("Invalid conv_type")

        self.act = nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
            self,
            feature_map,
            mask_map,
            position_map,
            condition_embedding,
            **kwargs,
    ):
        """
        :param self:
        :param feature_map: Tensor, shape = (B, C_in, H, W)
        :param mask_map: Tensor, shape = (B, 1, H, W)
        :param position_map: Tensor, shape = (B, 3, H, W)
        :return: Tensor, shape = (B, C_out, H, W)
        """
        if self.conv_type == "uv":
            shortcut = feature_map
            stats = self.adaLN_modulation(condition_embedding)
            shift_msa, scale_msa, gate_msa = stats
            shift_msa = shift_msa.unsqueeze(-1).unsqueeze(-1)
            scale_msa = scale_msa.unsqueeze(-1).unsqueeze(-1)
            gate_msa = gate_msa.unsqueeze(-1).unsqueeze(-1)

            # UV feature extraction
            feature_map = self.norm1(feature_map, mask_map)
            h = self.act(feature_map)
            h = self.conv_layer1(h)

            h = self.norm2(h, mask_map)
            h = h * (1.0 + scale_msa) + shift_msa
            h = self.act(h)
            h = self.dropout(h)
            uv_feature = self.conv_layer2(h)
            uv_feature = shortcut + uv_feature * gate_msa

            return uv_feature * mask_map

        elif self.conv_type == "point_uv":
            shortcut = feature_map

            stats = self.adaLN_modulation(condition_embedding)
            shift_msa, scale_msa, gate_msa = stats
            shift_msa = shift_msa.unsqueeze(-1).unsqueeze(-1)
            scale_msa = scale_msa.unsqueeze(-1).unsqueeze(-1)
            gate_msa = gate_msa.unsqueeze(-1).unsqueeze(-1)

            # UV feature extraction
            feature_map = self.norm1(feature_map, mask_map)
            h = self.act(feature_map)
            h = self.conv_layer1(h)

            h = self.norm2(h, mask_map)
            h = h * (1.0 + scale_msa) + shift_msa
            h = self.act(h)
            h = self.dropout(h)
            uv_feature = self.conv_layer2(h)

            # Point feature extraction
            point_stats = self.point_adaLN_modulation(condition_embedding)
            point_shift_msa, point_scale_msa, point_gate_msa = point_stats
            point_gate_msa = point_gate_msa.unsqueeze(-1).unsqueeze(-1)
            point_feature = self.extract_point_feature(
                shortcut, mask_map, position_map, point_shift_msa, point_scale_msa, self.voxel_size
            )

            fuse_feature = shortcut + uv_feature * gate_msa + point_feature * point_gate_msa

            return fuse_feature * mask_map

        elif self.conv_type == "point":
            shortcut = feature_map

            # Point feature extraction
            point_stats = self.point_adaLN_modulation(condition_embedding)
            point_shift_msa, point_scale_msa, point_gate_msa = point_stats
            point_gate_msa = point_gate_msa.unsqueeze(-1).unsqueeze(-1)
            point_feature = self.extract_point_feature(
                shortcut, mask_map, position_map, point_shift_msa, point_scale_msa, self.voxel_size
            )

            fuse_feature = shortcut + point_feature * point_gate_msa
            return fuse_feature * mask_map

        else:
            raise ValueError("Invalid conv_type")

    def extract_point_feature(self, feature_map, mask_map, position_map, shift_3d, scale_3d, voxel_size=0.01):
        # point_feature: Tensor, shape = (B, N, C_in)
        # point_mask: Tensor, shape = (B, N)
        # point_position: Tensor, shape = (B, N, 3)
        B, C, H, W = feature_map.shape
        raw_feats = rearrange(feature_map, "B C H W -> (B H W) C")
        position_map = rearrange(position_map, "B C H W -> B C (H W)")
        normalized_position = position_map - position_map.min(dim=2, keepdim=True)[0]
        raw_coords = rearrange(normalized_position, "B C N -> (B N) C")

        mask = rearrange(mask_map, 'B C H W -> (B H W) C').bool().squeeze(-1)
        # avoid empty mask!!!
        mask[0] = True
        mask = mask.detach()

        batch_id_map = torch.arange(B, device=mask.device).view(B, 1).expand(-1, H * W)
        batch_id_map = rearrange(batch_id_map, "B N -> (B N)").unsqueeze(-1)

        coords = torch.cat([batch_id_map, raw_coords], dim=1)[mask]
        features = raw_feats[mask]

        voxel_coords, voxel_feature_pool, idx_query = voxelize_with_feature_pool(coords, features, voxel_size)
        sparse_feat = SparseTensor(voxel_feature_pool, voxel_coords, stride=1)

        out_sparse_feat = self.point_block(sparse_feat, shift_3d, scale_3d)
        recon_feature = devoxelize_with_feature_nearest(out_sparse_feat.F, idx_query)

        inverse_point = torch.zeros_like(raw_feats, dtype=recon_feature.dtype)
        inverse_point[mask] = recon_feature

        out_map_feature = rearrange(inverse_point, "(B H W) C -> B C H W", B=B, H=H)

        return out_map_feature


class PixelNorm(nn.Module):
    def __init__(self, num_channels, layer_norm=False, affine=False):
        super(PixelNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = 1e-6
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nn.LayerNorm(num_channels, elementwise_affine=affine)

    def forward(self, input, mask):
        x = input * mask
        B, C, H, W = x.size()
        x = rearrange(x, "B C H W -> B (H W) C")

        if self.layer_norm:
            x_norm = self.norm(x)
        else:
            scale_x = (C ** (-0.5)) * x + self.eps
            x_norm = x / (scale_x.norm(dim=-1, keepdim=True))
        output = rearrange(x_norm, "B (H W) C -> B C H W", B=B, H=H)
        if torch.isnan(output).any():
            breakpoint()
        output = output * mask
        return output

class UVAdaLayerNormZero(nn.Module):
    def __init__(self, num_features, gate_num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(1024, 2 * num_features, bias=True)
        self.linear2 = nn.Linear(1024, gate_num_features, bias=True)

    def forward(self, condition_embedding):
        emb = self.linear1(self.act(condition_embedding))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        gate_msa = self.linear2(self.act(condition_embedding))

        return shift_msa, scale_msa, gate_msa

class UVPTVAttnStage(nn.Module):
    def __init__(self, in_channels, voxel_size, num_layers, num_heads, point_block_num, order, window_size, use_uv_head, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.uv_dit_blocks = nn.ModuleList()
        self.order = order
        self.shuffle_orders = True

        for i in range(num_layers):
            block = UV_DitBlock(in_channels, num_heads, point_block_num, order, window_size, use_uv_head)
            self.uv_dit_blocks.append(block)

    def forward(
            self,
            feature_map,
            mask_map,
            position_map,
            condition_embedding,
            **kwargs,
    ):
        point_feature = self.extract_feature(
            feature_map, mask_map, position_map, condition_embedding, self.voxel_size,
        )

        return point_feature * mask_map

    def extract_feature(
            self,
            feature_map,
            mask_map,
            position_map,
            condition_embedding,
            voxel_size=0.01
    ):
        # point_feature: Tensor, shape = (B, N, C_in)
        # point_mask: Tensor, shape = (B, N)
        # point_position: Tensor, shape = (B, N, 3)
        B, C, H, W = feature_map.shape
        re_position_map = rearrange(position_map, "B C H W -> B C (H W)")
        normalized_position = re_position_map - re_position_map.min(dim=2, keepdim=True)[0]
        raw_coords = rearrange(normalized_position, "B C N -> (B N) C")

        # avoid empty mask!!!
        mask = mask_map.bool()
        mask[:, :, 0, :] = True
        mask = rearrange(mask, 'B C H W -> (B H W) C').squeeze(-1)
        mask = mask.detach()

        batch_id_map = torch.arange(B, device=mask.device).view(B, 1).expand(-1, H * W)
        batch_id_map = rearrange(batch_id_map, "B N -> (B N)").unsqueeze(-1)

        coords = torch.cat([batch_id_map, raw_coords], dim=1)[mask]

        voxel_coords, idx_query = voxelize_without_feature_pool(coords, voxel_size)
        voxel_feature_pool = None

        data_dict = {
            "feat": voxel_feature_pool,
            "batch": voxel_coords[:, 0].long(),
            "grid_coord": voxel_coords[:, 1:],
            "condition_embedding": condition_embedding,
        }

        # core forward ----------------
        point = PTV3_Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

        for uv_dit_block in self.uv_dit_blocks:
            feature_map, point = uv_dit_block(
                feature_map, mask, mask_map, position_map, condition_embedding, point, idx_query)

        return feature_map

class UVBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super().__init__()
        self.conv_down = nn.Conv2d(in_channels + 3, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_up = nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.norm1 = PixelNorm(inter_channels, layer_norm=True, affine=True)
        self.norm2 = PixelNorm(inter_channels, layer_norm=True, affine=False)
        self.adaLN_modulation = UVAdaLayerNormZero(inter_channels, in_channels)
        self.conv_layer1 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_layer2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.act = nn.SiLU()
        self.norm_before_up = PixelNorm(inter_channels, layer_norm=True, affine=True)


    def forward(self, feature_map, mask_map, position_map, condition_embedding):
        shortcut = feature_map

        feature_map = torch.cat([feature_map, position_map], dim=1)
        feature_map = self.conv_down(feature_map)

        # UV feature extraction
        feature_map = self.norm1(feature_map, mask_map)
        stats = self.adaLN_modulation(condition_embedding)
        shift_msa, scale_msa, gate_msa = stats
        shift_msa = shift_msa.unsqueeze(-1).unsqueeze(-1)
        scale_msa = scale_msa.unsqueeze(-1).unsqueeze(-1)
        gate_msa = gate_msa.unsqueeze(-1).unsqueeze(-1)

        h = self.act(feature_map)
        h = self.conv_layer1(h)

        h = self.norm2(h, mask_map)
        h = h * (1.0 + scale_msa) + shift_msa
        h = self.act(h)
        uv_feature = self.conv_layer2(h)
        uv_feature = self.norm_before_up(uv_feature, mask_map)
        uv_feature = self.act(uv_feature)
        uv_feature = self.conv_up(uv_feature)

        uv_feature = shortcut + uv_feature * gate_msa

        return uv_feature * mask_map

class PointBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.act_layer = nn.SiLU()
        self.conv1 = spnn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, dilation=1, bias=True)
        self.conv2 = spnn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, dilation=2, bias=True)
        self.norm1 = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(in_dim, elementwise_affine=False)

    def forward(self, point, shift_3d, scale_3d):
        batch_id = point.coords[:, 0]
        shift_3d = shift_3d[batch_id]
        scale_3d = scale_3d[batch_id]

        point.F = point.F.float()
        point.F = self.norm1(point.F)
        point.F = self.act_layer(point.F)
        point = self.conv1(point)

        point.F = self.norm2(point.F)
        point.F = point.F * (1.0 + scale_3d) + shift_3d
        point.F = self.act_layer(point.F)
        point = self.conv2(point)

        return point

class UV_DitBlock(nn.Module):
    def __init__(self, in_channels, num_heads, point_block_num, order, window_size, use_uv_head):
        super().__init__()
        self.use_uv_head = use_uv_head
        if use_uv_head:
            self.uv_head = UVBlock(in_channels, inter_channels=256)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = True

        self.dit_blocks = PointSequential()

        for i in range(point_block_num):
            self.dit_blocks.add(
                TimeBlock(
                    channels=in_channels,
                    num_heads=num_heads,
                    patch_size=window_size,
                    qkv_bias=True,
                    drop_path=0.3,
                    order_index=i % len(self.order),
                    enable_flash=True,
                    upcast_attention=False,
                    upcast_softmax=False,
                    qk_norm=True,
                    use_cpe=True,
                    cond_embed_dim=1024,
                ),
            )

        self.point_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1024, in_channels, bias=True),
        )

    def forward(
            self,
            feature_map,
            mask,
            mask_map,
            position_map,
            condition_embedding,
            point,
            idx_query,
    ):
        B, C, H, W = feature_map.shape
        gate_point = self.point_gate(condition_embedding).unsqueeze(-1).unsqueeze(-1)

        if self.use_uv_head:
            feature_map = self.uv_head(feature_map, mask_map, position_map, condition_embedding)

        shortcut = feature_map
        raw_feats = rearrange(feature_map, "B C H W -> (B H W) C")
        features = raw_feats[mask]
        voxel_feature_pool = torch_scatter.scatter_mean(features, idx_query.long(), dim=0)
        point.feat = voxel_feature_pool

        point.sparsify()
        point = self.dit_blocks(point)

        recon_feature = devoxelize_with_feature_nearest(point.feat, idx_query)
        inverse_point = torch.zeros_like(raw_feats, dtype=recon_feature.dtype)
        inverse_point[mask] = recon_feature

        feature_map = rearrange(inverse_point, "(B H W) C -> B C H W", B=B, H=H) * gate_point + shortcut

        return feature_map, point