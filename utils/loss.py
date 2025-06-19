import torch.nn as nn
from torch.nn import functional as F

from utils.perceptual import VGGPerceptualLoss, perceptual_loss

class Loss_Func():
    def __init__(self, perceptual_weight, bg, device):
        self.perceptual_weight = perceptual_weight
        self.bg = bg
        self.vgg_loss = VGGPerceptualLoss(resize=False).to(device)
    
    def forward(self, model_out, uv_maps, ctx, mesh, eval_mvps):
        return perceptual_loss(self.vgg_loss, model_out, uv_maps, ctx, mesh, eval_mvps, ) * self.perceptual_weight

