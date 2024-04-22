# coding=utf-8
# @FileName:VSSBlock.py
# @Time:2024/2/20 
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

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from torchvision.ops import Permute

from models.utils import PatchEmbed
from models.vmamba import SS2D, Mlp


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 96,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=0.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            print("mlp_hidden_dim", mlp_hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class vision_mamba(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans,
                 embed_dim,
                 pre_norm=False,
                 class_token=True,
                 no_embed_class=False,
                 norm_layer=None,
                 embed_layer=PatchEmbed,
                 depth=12,
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 num_classes=43,
                 drop_rate=0.,
                 ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        num_patches = self.patch_embed.num_patches
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.blocks = nn.Sequential(*[
            VSSBlock(
                hidden_dim=embed_dim,
                drop_path=0.1,
                use_checkpoint=False,
                norm_layer=nn.LayerNorm,
                downsample=nn.Identity(),
                # ===========================
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_rank_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0,
                ssm_simple_init=False,
                forward_type="v2",
                # ===========================
                mlp_ratio=0.0,
                mlp_act_layer=nn.GELU,
                mlp_drop_rate=0.0,
            )
            for i in range(depth)])

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        print(x.shape)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


if __name__ == '__main__':
    """
        torch.Size([64, 16, 192])
        patchsize 大小为8
        共有 4*4 = 16 个 patch
        每个patchd的大小为 8 * 8 * 3 = 192
        """
    input = torch.randn(64, 3, 32, 32)
    input = input.permute(0, 2, 3, 1)
    print("input.shape", input.shape)
    input = input.cuda()
    model = VSSBlock(hidden_dim=3)
    model = model.cuda()
    output = model(input)
    print(output.shape)
