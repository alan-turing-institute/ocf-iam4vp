"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import torch
from torch import nn

from .modules import Attention, ConvNeXt_block, ConvNeXt_bottle, ConvSC, TimeMLP


def stride_generator(N, reverse=False):
    strides = [1, 2] * N
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    """
    Spatial encoder

    Transform data from the full phase space into a reduced latent space
    """

    def __init__(self, C_in, C_hid, N_S):
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
    """
    LearnedPrior

    Transform priors from the full phase space into a reduced latent space
    """

    def __init__(self, C_in, C_hid, N_S):
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
    """
    Decoder

    Transform data from the reduced latent space into the full phase space
    """

    def __init__(self, C_hid, C_out, N_S):
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
    """
    Spatio-temporal predictor

    Predict in latent space using ConvNeXt blocks
    """

    def __init__(self, channel_in, channel_hid, N_T):

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
    """
    IAM4VP model

    - Spatial encoder to a latent phase space
    - Combine with future frame information
    - Combine with sinusoidal time MLP
    - Run spatio-temporal predictor
    - Spatial decoder to original phase space
    - Spatial temporal refinement
    """

    def __init__(self, shape_in, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        T, C, H, W = shape_in
        self.time_mlp = TimeMLP(dim=hid_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.attn = Attention(C * T)
        self.readout = nn.Conv2d(C * T, C, 1)
        self.mask_token = nn.Parameter(
            torch.zeros(T, hid_S, -(H // -N_S), -(W // -N_S))
        )
        self.lp = LearnedPrior(C, hid_S, N_S)

    def forward(self, x_raw, y_raw=None, t=None):
        """
        Transformation summary

        Inputs:
            x: (batch_size, history_steps, channels, height, width)
            y_raw: N * (batch_size, channels, height, width)

        Outputs:
            (batch_size, channels, height, width)
        """

        # Combine batch and time information
        B, T, C, H, W = x_raw.shape
        x = x_raw.contiguous().view(B * T, C, H, W)

        # Encode to latent space
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # Embed future frames via learned prior
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)
        for idx, pred in enumerate(y_raw):
            embed_priors, _ = self.lp(pred)
            mask_token[:, idx, :, :, :] = embed_priors

        # Combine data and priors in latent space
        x_latent = embed.view(B, T, C_, H_, W_)
        priors_latent = mask_token
        combined_latent = torch.cat([x_latent, priors_latent], dim=1)

        # Run predictor on combined latent data + time embedding
        time_emb = self.time_mlp(t)
        hid = self.hid(combined_latent, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)

        # Decode the output
        Y = self.dec(hid, skip)

        # Move the history steps dimension onto channels
        Y = Y.reshape(B, C * T, H, W)

        # Run the attention step
        Y = self.attn(Y)

        # Readout to the correct shape
        Y = self.readout(Y)
        return Y
