# coding=utf-8
# @FileName:MambaReID.py
# @Time:2024/2/22 
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
import copy
from models import build_model
from test_param import tmp_parse_option


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SMBlock(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.model = build_model(setting)

    def forward(self, x):
        x = self.model(x)
        return x

    def load_param(self, trained_path):
        checkpoint = torch.load(trained_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print("loading weights done !")


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


class MambaReID(nn.Module):
    def __init__(self, setting, trained_path, depth=12):
        super().__init__()
        self.in_chans = 3
        self.embed_dim = 128
        norm_layer = nn.LayerNorm

        self.prembed = nn.Sequential(
            nn.Conv2d(self.in_chans, self.embed_dim // 2, kernel_size=3, stride=1, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(self.embed_dim // 2),
            Permute(0, 3, 1, 2),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, kernel_size=3, stride=1, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(self.embed_dim),
        )
        # 获取 swin mamba 模型 并加载权重
        self.base = SMBlock(setting)
        self.base.load_param(trained_path)

        # 针对不同版本 确定每个blockd的数目
        self.block_nums = depth // 4
        self.dims = [128, 256, 512, 1024, 2048]
        self.layers = nn.ModuleList()

        source_layers = [
            copy.deepcopy(self.base.model.layers[i].blocks[1]) for i in range(4)
        ]

        down_sampling_layers = [PatchMerging2D(
            i_layer,
            i_layer * 2,
            norm_layer=nn.LayerNorm,
        ) for i_layer in self.dims if i_layer < 2048]

        self.mr_layer_0 = nn.Sequential(*[copy.deepcopy(source_layers[0]) for _ in range(self.block_nums)])
        self.dw0 = down_sampling_layers[0]

        self.mr_layer_1 = nn.Sequential(*[copy.deepcopy(source_layers[1]) for _ in range(self.block_nums)])
        self.dw1 = down_sampling_layers[1]

        self.mr_layer_2 = nn.Sequential(*[copy.deepcopy(source_layers[2]) for _ in range(self.block_nums)])
        self.dw2 = down_sampling_layers[2]

        self.mr_layer_3 = nn.Sequential(*[copy.deepcopy(source_layers[3]) for _ in range(self.block_nums)])
        self.dw3 = down_sampling_layers[3]

        self.conv1 = nn.Conv2d(2048, 768, kernel_size=1)


    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.prembed(x)

        x = self.mr_layer_0(x)
        x = self.dw0(x)

        x = self.mr_layer_1(x)
        x = self.dw1(x)

        x = self.mr_layer_2(x)
        x = self.dw2(x)

        x = self.mr_layer_3(x)
        x = self.dw3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, 128, 768)
        return x


if __name__ == '__main__':
    _, setting = tmp_parse_option()
    # print(setting.MODEL.PRETRAINED)
    model = MambaReID(setting, setting.MODEL.PRETRAINED)
    # model = SMBlock(setting)
    # print(model)
    model = model.cuda()
    input = torch.randn(1, 3, 256, 128)
    input = input.cuda()
    output = model(input)
    print(output.shape)
