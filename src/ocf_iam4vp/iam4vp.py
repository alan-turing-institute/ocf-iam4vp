"""Implementation of IAM4VP from https://github.com/seominseok0429/Implicit-Stacked-Autoregressive-Model-for-Video-Prediction"""

import numpy as np
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


class LearnedPrior(nn.Module):
    """
    LearnedPrior

    Transform priors from the full phase space into a reduced latent space
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

    Parameters:
    - hid_S: number of spatial hidden channels
    - N_S: number of spatial convolution layers
    - N_T: number of temporal convolution layers
    """

    def __init__(
        self,
        shape_in: torch.Size,
        hid_S: int = 64,
        hid_T: int = 512,
        N_S: int = 4,
        N_T: int = 6,
    ):
        super().__init__()
        T, C, H, W = shape_in
        self.time_mlp = TimeMLP(dim=hid_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T, hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.mask_token = nn.Parameter(
            torch.zeros_like(self.enc(torch.randn(shape_in))[0])
        )
        self.lp = LearnedPrior(C, hid_S, N_S)
        self.str = SpatioTemporalRefinement(C, T)

    def forward(
        self,
        x_raw: torch.Tensor,
        y_raw: list[torch.Tensor] = [],
        t_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Transformation summary

        Inputs:
            x: (batch_size, history_steps, channels, height, width)
            y_raw: N * (batch_size, channels, height, width)
            t: (batch_size)

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

        # Construct time embedding
        times = t_raw if t_raw else torch.tensor(100).repeat(B).to(x.device)
        time_emb = self.time_mlp(times)

        # Run predictor on combined latent data + time embedding
        hid = self.hid(combined_latent, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)

        # Decode the output
        Y = self.dec(hid, skip)

        # Perform STR and return
        return self.str(Y)

    def predict(
        self,
        X: np.ndarray,
        n_forecast_steps: int,
        device: str,
    ) -> torch.Tensor:
        """
        Make predictions with a trained model

        Inputs:
            X [numpy array]: (batch_size, channels, time, height, width)
            n_forecast_steps [int]: number of steps to forecast
            device [str]: Which PyTorch device to use

        Outputs:
            y_hat [numpy array]: (batch_size, channels, n_forecast_steps, height, width)
        """
        # Disable gradient calculation in evaluate mode
        with torch.no_grad():

            # Load data into tensor with shape (batch_size, time, channels, height, width)
            batch_X = torch.from_numpy(X).swapaxes(1, 2).to(device)

            # Generate the requested number of forecasts
            y_hats: list[torch.Tensor] = []
            for f_step in range(n_forecast_steps):

                # Generate an appropriately-sized set of blank times
                times = torch.tensor(f_step * 100).repeat(batch_X.shape[0]).to(device)

                # Forward pass for the next time step
                y_hat: torch.Tensor = self(batch_X, y_hats, times)

                # Store the prediction
                # Note that y_hat has shape (batch_size, channels, height, width)
                y_hats.append(y_hat.detach())

                # Cleanup loop variables
                del times

            # Cleanup inputs
            del batch_X

            # Convert results to the expected output format by doing the following:
            # - Add a time axis
            # - convert from Tensor to numpy array
            # - concatenate the forecasts along the time axis
            # - ensure data is in the range (0, 1) or -1
            y_hat_np = [y_hat[:, :, None, :, :].cpu().numpy() for y_hat in y_hats]
            y_hat_concat = np.concatenate(y_hat_np, axis=2)
            y_hat_concat[y_hat_concat < 0] = -1
            y_hat_concat[y_hat_concat > 1] = 1
            y_hat_concat = np.nan_to_num(y_hat_concat, nan=-1, posinf=1)

            # Cleanup predictions
            for y_hat in y_hats:
                del y_hat

        # Return concatenated output
        return y_hat_concat
