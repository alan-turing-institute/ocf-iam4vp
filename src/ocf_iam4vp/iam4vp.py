"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import math

import torch
from torch import nn

from .modules import Attention, ConvNeXt_block, ConvNeXt_bottle, ConvSC


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        """Encode data from the full phase space into a reduced latent space"""
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        """
        Transformation summary

        Inputs:
            x: (batch_size * history_steps, channels, height, width)

        Outputs:
            latent: (batch_size, hidden_spatial, height_latent, width_latent)
            enc1: (batch_size, hidden_spatial, height, width)
        """
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class LearnedPrior(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        """Encode learned prior from the full phase space into a reduced latent space"""
        super(LearnedPrior, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        """
        Transformation summary

        Inputs:
            x: (batch_size, channels, height, width)

        Outputs:
            latent: (batch_size, hidden_spatial, height_latent, width_latent)
            enc1: (batch_size, hidden_spatial, height, width)
        """
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        """Decode data from the reduced latent space into the full phase space"""
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        """
        Transformation summary

        Inputs:
            hid: (batch_size * history_steps, hidden_spatial, height_latent, width_latent)
            enc1: (batch_size * history_steps, hidden_spatial, height, width)

        Outputs:
            (batch_size * history_steps, channels, height, width)
        """
        # Deconvolve hid back to full size
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        # Crop hid to the same size as enc1
        hid = hid[:, :, : enc1.shape[-2], : enc1.shape[-1]]
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        """Predict in latent space using ConvNeXt blocks"""
        super(Predictor, self).__init__()
        self.st_block = nn.Sequential(
            ConvNeXt_bottle(dim=channel_in),
            *[ConvNeXt_block(dim=channel_in) for _ in range(N_T)],
        )

    def forward(self, x, time_emb):
        """
        Transformation summary

        Inputs:
            x: (batch_size, history_steps * 2, hidden_spatial, height_latent, width_latent)
            time_emb: (batch_size, hidden_spatial)
        Outputs:
            (batch_size, history_steps, hidden_spatial, height_latent, width_latent)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, len(self.st_block)):
            z = self.st_block[i](z, time_emb)
        y = z.reshape(B, -1, C, H, W)
        return y


class IAM4VP(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        T, C, H, W = shape_in
        self.time_mlp = Time_MLP(dim=64)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.attn = Attention(64)
        self.readout = nn.Conv2d(64, 1, 1)
        self.mask_token = nn.Parameter(torch.zeros(10, hid_S, 16, 16))
        self.lp = LearnedPrior(C, hid_S, N_S)

    def forward(self, x_raw, y_raw=None, t=None):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        time_emb = self.time_mlp(t)
        embed, skip = self.enc(x)
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        for idx, pred in enumerate(y_raw):
            embed2, _ = self.lp(pred)
            mask_token[:, idx, :, :, :] = embed2

        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = self.attn(Y)
        Y = self.readout(Y)
        return Y
