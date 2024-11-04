"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from torch import nn


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        Transformation summary

        Inputs:
            x: (batch_size, hidden_spatial * history_steps, height_latent, width_latent)
            time_emb: (batch_size, hidden_spatial)

        Outputs:
            (batch_size, hidden_spatial * history_steps, height_latent, width_latent)
        """
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BasicConv2d(nn.Module):
    """Basic 2D convolutional layer from https://github.com/A4Bio/SimVP"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        transpose=False,
        act_norm=False,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if transpose is True:
            self.conv = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels,
                        out_channels * 4,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Transformation summary

        Inputs:
            x: (batch_size * history_steps, channels_in, height, width)

        Outputs:
            (batch_size * history_steps, channels_out, height, width)
        """
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """Basic 2D convolutional layer from https://github.com/A4Bio/SimVP"""

    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm,
        )

    def forward(self, x):
        """
        Transformation summary

        Inputs:
            x: (batch_size * history_steps, channels_in, height, width)

        Outputs:
            (batch_size * history_steps, channels_out, height, width)
        """
        return self.conv(x)


class ConvNeXt_block(nn.Module):
    """
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(64, dim))
        self.dwconv = LKA(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Transformation summary

        Inputs:
            x: (batch_size, hidden_spatial * history_steps, height_latent, width_latent)
            time_emb: (batch_size, hidden_spatial)

        Outputs:
            (batch_size, hidden_spatial * history_steps, height_latent, width_latent)
        """
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) + rearrange(time_emb, "b c -> b c 1 1")
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt_bottle(nn.Module):
    """
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(64, dim))
        self.dwconv = nn.Conv2d(
            dim * 2, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.res_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, time_emb=None):
        """
        Transformation summary

        Inputs:
            x: (batch_size, 2 * hidden_spatial * history_steps, height_latent, width_latent)
            time_emb: (batch_size, hidden_spatial)

        Outputs:
            (batch_size, hidden_spatial * history_steps, height_latent, width_latent)
        """
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) + rearrange(time_emb, "b c -> b c 1 1")
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.res_conv(input) + self.drop_path(x)
        return x


class LKA(nn.Module):
    """Large Kernel Attention from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Transformation summary

        Inputs:
            x: (batch_size, channel_dim, height_dim, width_dim)

        Outputs:
            (batch_size, channel_dim, height_dim, width_dim)
        """
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Attention(nn.Module):
    """Attention from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        """
        Transformation summary

        Inputs:
            x: (batch_size, channels * history_steps, height, width)

        Outputs:
            (batch_size, channels * history_steps, height, width)
        """
        initial_input = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + initial_input
        return x


class Bottleneck(nn.Module):
    """Bottleneck module"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Args:
            inplanes (int): no. input channels
            planes (int): no. output channels
            stride (int): stride
            downsample (nn.Module): downsample module
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = LayerNorm(planes, eps=1e-6, data_format="channels_first")
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = LayerNorm(
            planes, eps=1e-6, data_format="channels_first"
        )  # planes * self.expansion
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = LayerNorm(
            planes * self.expansion, eps=1e-6, data_format="channels_first"
        )
        self.relu = nn.SiLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
