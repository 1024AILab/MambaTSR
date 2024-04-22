import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from vmamba import SS2D, Mlp
from functools import partial
from typing import Optional, Callable, Any
import torch
from timm.models.layers import DropPath, trunc_normal_


class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, H, W, C = x.size()
        x = x.view(B, -1, C)
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(B, H, W, C)
        return x

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

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 16, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 16, 1, kernel_size=1)
        )

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            print("mlp_hidden_dim", mlp_hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)
        self.norm3 = norm_layer(hidden_dim)
        self.sgfn = SGFN(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(0.)

    def _forward(self, input: torch.Tensor):
        x_norm = self.norm(input)
        vss = self.op(x_norm)

        # 调整通道顺序变为 [b, c, h, w]
        x_norm_after_LN = x_norm.permute(0, 3, 1, 2)
        conv_x = self.dwconv(x_norm_after_LN)
        channel_map = self.channel_interaction(conv_x).contiguous()

        # 调整通道顺序变为 [b, c, h, w]
        vss_x = vss.permute(0, 3, 1, 2)
        attention_reshape = vss_x.contiguous()
        spatial_map = self.spatial_interaction(attention_reshape)
        attened_x = vss_x * torch.sigmoid(channel_map)
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.contiguous()

        # dynamic interaction block
        x = attened_x + conv_x
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + x_norm

        # Spatial-Gate Feed-Forward Network
        norm_before_sgfn = self.norm3(x)
        sgfn = self.sgfn(norm_before_sgfn)
        out_f = sgfn + norm_before_sgfn

        x = out_f

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

        
if __name__ == '__main__':
    device = torch.device("cuda:0")
    input = torch.randn(2, 256, 256, 32)
    input = input.to(device)
    model = VSSBlock(hidden_dim=32)
    model = model.to(device)
    output = model(input)
    print(output.shape)