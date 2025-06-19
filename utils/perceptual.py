import random
import math
import torch
import torchvision

from spuv.nvdiffrast_utils import render_rgb_from_texture_mesh
from spuv.ops import get_projection_matrix, get_mvp_matrix
from spuv.camera import get_c2w

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, input_range=(0, 1), feature_layers=[2,3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if input_range == (0, 1):
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        else:
            raise NotImplementedError
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.mean(torch.nn.functional.l1_loss(x, y, reduction="none"), dim=[1, 2, 3])
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.mean(torch.nn.functional.l1_loss(gram_x, gram_y, reduction="none"), dim=[1, 2, 3])
        return loss.unsqueeze(1).unsqueeze(2).unsqueeze(3)

def perceptual_loss(vgg_loss, input_uv_map, uv_map, ctx, meshes, mvps, bg):
    input = []
    target = []
    for i, mesh in enumerate(meshes):
        input_rgbs = render_rgb_from_texture_mesh(ctx, mesh, input_uv_map[i], mvps, 224, 224, bg)
        target_rgbs = render_rgb_from_texture_mesh(ctx, mesh, uv_map[i], mvps, 224, 224, bg)
        input.append(input_rgbs)
        target.append(target_rgbs)
    
    input = torch.concat(input, dim=0).permute(0, 3, 1, 2)
    target = torch.concat(target, dim=0).permute(0, 3, 1, 2)
    
    loss = vgg_loss(input, target)
    loss = torch.mean(loss)
    
    return loss

def get_rand_mvp(num, ele_range, camera_dist, fovy):
    eles = torch.tensor([random.uniform(ele_range[0], ele_range[1]) for _ in range(num)])
    azims = torch.tensor([random.uniform(0, 360) for _ in range(num)])

    c2w = get_c2w(azims, eles, camera_dist)
    proj = get_projection_matrix(fovy, 1, 0.1, 1000.0)
    mvp = get_mvp_matrix(c2w, proj)

    return mvp