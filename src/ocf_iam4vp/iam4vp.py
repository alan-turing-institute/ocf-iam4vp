"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import torch
from torch import nn

from .modules import Attention, ConvNextTimeEmbedLKA, ConvNextTimeEmbed, ConvSC, TimeMLP


def stride_generator(N: int, reverse=False) -> list[int]:
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

    def __init__(self, C_in: int, C_hid: int, N_S: int) -> None:
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class Decoder(nn.Module):
    """
    Decoder

    Transform data from the reduced latent space into the full phase space
    """

    def __init__(self, C_hid: int, C_out: int, N_S: int) -> None:
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(
        self, hid: torch.Tensor, enc1: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def __init__(self, history_steps: int, hid_S: int, hid_T: int, N_T: int) -> None:
        super().__init__()
        self.spatial_to_temporal = nn.Conv2d(2 * history_steps * hid_S, hid_T, 1)
        self.cn_blocks = nn.Sequential(
            ConvNextTimeEmbed(dim=hid_T, dim_time_embed=hid_S),
            *[
                ConvNextTimeEmbedLKA(dim=hid_T, dim_time_embed=hid_S)
                for _ in range(N_T - 1)
            ],
        )
        self.temporal_to_spatial = nn.Conv2d(hid_T, history_steps * hid_S, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
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
        x = self.spatial_to_temporal(x)
        x = self.cn_blocks[0](x, time_emb)
        for i in range(1, len(self.cn_blocks)):
            x = self.cn_blocks[i](x, time_emb)
        x = self.temporal_to_spatial(x)
        x = x.reshape(B, -1, C, H, W)
        return x


class SpatioTemporalRefinement(nn.Module):
    """
    Spatio-temporal refinement

    Refine prediction in original phase space
    """

    def __init__(self, channels: int, history_steps: int) -> None:
        super().__init__()
        self.channels_in = channels * history_steps
        self.attn = Attention(self.channels_in)
        self.readout = nn.Conv2d(self.channels_in, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size * history_steps, channels, height, width)
        Outputs:
            (batch_size, 1, height, width)
        """
        _, _, H, W = x.shape

        # Move the history steps dimension onto channels
        x = x.reshape(-1, self.channels_in, H, W)

        # Run the attention step
        x = self.attn(x)

        # Readout to the correct shape
        return self.readout(x)


class IAM4VP(nn.Module):
    """
    IAM4VP model

    - Spatial encoder to a latent phase space
    - Combine with future frame information
    - Combine with sinusoidal time MLP
    - Run spatio-temporal predictor
    - Spatial decoder to original phase space
    - Spatial temporal refinement (STR)
    - Sigmoid normalisation to range (0, 1)

    Parameters:
    - num_forecast_steps [int]: number of steps to forecast
    - hid_S [int]: number of spatial hidden channels
    - hid_T [int]: number of temporal hidden channels
    - N_S [int]: number of spatial convolution layers
    - N_T [int]: number of temporal convolution layers
    """

    def __init__(
        self,
        shape_in: torch.Size,
        num_forecast_steps: int = 12,
        hid_S: int = 64,
        hid_T: int = 512,
        N_S: int = 4,
        N_T: int = 6,
    ):
        super().__init__()
        T, C, H, W = shape_in
        self.num_forecast_steps = num_forecast_steps
        self.time_mlp = TimeMLP(dim=hid_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T, hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.future_latent = nn.Parameter(
            torch.zeros_like(self.enc(torch.randn(shape_in))[0])
        )
        self.str = SpatioTemporalRefinement(C, T)
        self.norm = torch.nn.Sigmoid()

    def forward(
        self,
        x_raw: torch.Tensor,
        y_raw: list[torch.Tensor] = [],
    ) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size, channels, history_steps, height, width)
            y_raw: N * (batch_size, channels, height, width)

        Outputs:
            (batch_size, channels, height, width)
        """
        # Convert from (B, C, T, H, W) to (B * T, C, H, W)
        B, C, T, H, W = x_raw.shape
        x = x_raw.swapaxes(1, 2).contiguous().view(B * T, C, H, W)

        # Encode input to latent space
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        # Encode future frames to latent space, padding with zeros
        future_latent = self.future_latent.repeat(B, 1, 1, 1, 1)
        for idx, pred in enumerate(y_raw):
            pred_embed, _ = self.enc(pred)
            future_latent[:, idx, :, :, :] = pred_embed
            del pred_embed

        # Combine data and priors in latent space
        context_latent = embed.view(B, T, C_, H_, W_)
        combined_latent = torch.cat([context_latent, future_latent], dim=1)
        del context_latent, future_latent, embed

        # Construct time embedding from a uniformly-filled tensor
        time_emb = self.time_mlp(torch.tensor(100 * len(y_raw)).repeat(B).to(x.device))

        # Run predictor on combined latent data + time embedding
        Y = self.hid(combined_latent, time_emb)
        Y = Y.reshape(B * T, C_, H_, W_)
        del combined_latent, time_emb

        # Decode the output
        Y = self.dec(Y, skip)
        del skip

        # Perform spatio-temporal refinement
        Y = self.str(Y)

        # Apply a sigmoid to restrict output to range (0, 1)
        Y = self.norm(Y)
        return Y

    def predict(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make predictions with a trained model

        Inputs:
            X [torch.Tensor]: (batch_size, channels, time, height, width)

        Outputs:
            y_hat [torch.Tensor]: (batch_size, channels, num_forecast_steps, height, width)
        """
        # Disable gradient calculation in evaluate mode
        with torch.no_grad():

            # Load data into tensor with shape (batch_size, time, channels, height, width)
            # Explicitly remove NaNs from input
            batch_X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)

            # Generate the requested number of forecasts
            # This gives a list of N tensors with shape (B, C, H, W)
            y_hats: list[torch.Tensor] = []
            for _ in range(self.num_forecast_steps):
                # Forward pass for the next time step
                # We append each prediction to the list of future frames
                y_hats.append(self.forward(batch_X, y_hats).detach())

            # Free up memory
            del batch_X

            # Convert results to the expected output format by doing the following:
            # - concatenate the forecasts along a new time axis
            # - ensure forecasts are in the range (0, 1)
            # - replace any NaNs or infinities with 0
            forecasts = torch.stack(y_hats, dim=2)
            forecasts = torch.clamp(forecasts, min=0, max=1)
            forecasts = torch.nan_to_num(forecasts, nan=0, posinf=0, neginf=0)
            del y_hats

        # Return tensor with shape (B, C, T, H, W)
        return forecasts
