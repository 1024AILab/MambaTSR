# coding=utf-8
# @FileName:VSSBlock_utils.py
# @Time:2024/2/21 
# @Author: CZH

import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from models.ConvNet import ConvNet
from models.VSSBlock import VSSBlock


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Super_Mamba(nn.Module):
    def __init__(self, dims=3, depth=6, num_classes=43):
        super().__init__()
        self.depth = depth
        self.preembd = ConvNet()
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.depth+1)]
        self.num_features = dims[-1]
        self.dims = dims
        self.layers = nn.ModuleList()
        for i_layer in range(self.depth):
            downsample = PatchMerging2D(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=nn.LayerNorm,
            )
            vss_block = VSSBlock(hidden_dim=self.dims[i_layer+1])
            self.layers.append(downsample)
            self.layers.append(vss_block)

        self.classifier = nn.Sequential(OrderedDict(
            norm=nn.LayerNorm(self.num_features),  # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.preembd(x)
        x = x.permute(0, 2, 3, 1)
        for layers in self.layers:
            x = layers(x)
        # print(x.shape)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dim = 3
    input = torch.randn(64, 3, 32, 32)
    input = input.permute(0, 2, 3, 1)
    print("input.shape", input.shape)
    input = input.cuda()
    model = PatchMerging2D(3, 6)
    model = model.cuda()
    output = model(input)
    print(output.shape)
