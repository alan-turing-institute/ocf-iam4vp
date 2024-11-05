"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import math

import torch
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn
from transformers.models.convnext.modeling_convnext import (
    ConvNextConfig,
    ConvNextLayer,
    ConvNextLayerNorm,
)


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding

    from https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    def __init__(self, dim: int, theta: float = 10000) -> None:
        super().__init__()
        self.half_dim = dim // 2
        self.scale = math.log(theta) / (self.half_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size)

        Outputs:
            (batch_size, hidden_spatial)
        """
        device = x.device
        emb = torch.exp(torch.arange(self.half_dim, device=device) * -self.scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeMLP(nn.Module):
    """
    Time multilayer perceptron

    from https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size)

        Outputs:
            (batch_size, hidden_spatial)
        """
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class BasicConv2d(nn.Module):
    """
    Basic 2D convolutional layer

    from https://github.com/A4Bio/SimVP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        transpose: bool = False,
        act_norm: bool = False,
    ) -> None:
        super().__init__()
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
        self.norm = ConvNextLayerNorm(
            out_channels, eps=1e-6, data_format="channels_first"
        )
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Basic 2D convolutional layer

    from https://github.com/A4Bio/SimVP
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        transpose: bool = False,
        act_norm: bool = True,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size * history_steps, channels_in, height, width)

        Outputs:
            (batch_size * history_steps, channels_out, height, width)
        """
        return self.conv(x)


class ConvNextBase(nn.Module):
    """
    The ConvNeXt block with the depthwise convolution and recombination steps dropped

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cnl = ConvNextLayer(
            ConvNextConfig(
                hidden_act="gelu", layer_scale_init_value=layer_scale_init_value
            ),
            dim=dim,
            drop_path=drop_path,
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size, channels, height, width)

        Outputs:
            (batch_size, channels, height, width)
        """
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.cnl.layernorm(x)
        x = self.cnl.pwconv1(x)
        x = self.cnl.act(x)
        x = self.cnl.pwconv2(x)
        if self.cnl.layer_scale_parameter is not None:
            x = self.cnl.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ConvNextTimeEmbedLKA(nn.Module):
    """
    ConvNeXt block, with LKA instead of depth-wise convolution and adapted to add time embeddings.

    from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction
    """

    def __init__(
        self,
        dim: int,
        hidden_spatial: int = 64,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_spatial, dim))
        self.dwconv = LargeKernelAttention(dim)
        self.convnext = ConvNextLayer(
            ConvNextConfig(
                hidden_act="gelu", layer_scale_init_value=layer_scale_init_value
            ),
            dim=dim,
            drop_path=drop_path,
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        x = self.convnext.layernorm(x)
        x = self.convnext.pwconv1(x)
        x = self.convnext.act(x)
        x = self.convnext.pwconv2(x)
        if self.convnext.layer_scale_parameter is not None:
            x = self.convnext.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.convnext.drop_path(x)
        return x


class ConvNextTimeEmbed(nn.Module):
    """
    ConvNeXt block, adapted to add time embeddings.

    from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction
    """

    def __init__(
        self,
        dim: int,
        hidden_spatial: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_spatial, dim))
        self.convnext = ConvNextLayer(
            ConvNextConfig(
                hidden_act="gelu", layer_scale_init_value=layer_scale_init_value
            ),
            dim=dim,
            drop_path=drop_path,
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        x = self.convnext.dwconv(x) + rearrange(time_emb, "b c -> b c 1 1")
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.convnext.layernorm(x)
        x = self.convnext.pwconv1(x)
        x = self.convnext.act(x)
        x = self.convnext.pwconv2(x)
        if self.convnext.layer_scale_parameter is not None:
            x = self.convnext.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.convnext.drop_path(x)
        return x


class LargeKernelAttention(nn.Module):
    """Large Kernel Attention from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LargeKernelAttention(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
