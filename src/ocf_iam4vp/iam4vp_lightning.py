import torch
import torch.optim as optim
import lightning as L
from tqdm import tqdm
from typing import Any

from .iam4vp import IAM4VP


class IAM4VPLightning(L.LightningModule):
    """
    IAM4VP lightning model

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
        self.save_hyperparameters()
        self.model = IAM4VP(shape_in, num_forecast_steps, hid_S, hid_T, N_S, N_T)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        # Split the batch into X and y
        batch_X, batch_y = batch

        # Generate the requested number of forecasts
        y_hats: list[torch.Tensor] = []
        single_forecast_losses = []
        for idx_forecast in range(self.model.num_forecast_steps):
            # Forward pass for the next time step (batch_size, channels, height, width)
            y_hat = self.model(batch_X, y_hats)

            # Calculate the loss
            y = batch_y[:, :, idx_forecast, :, :]
            single_forecast_losses.append(self.loss(y_hat, y))
            del y

            # Append latest prediction to queue
            y_hats.append(y_hat.detach())

        # Calculate mean loss for the full set of forecasts
        loss = torch.stack(single_forecast_losses).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Split the batch into X and y
        batch_X, batch_y = batch

        # Generate a prediction
        batch_y_hat = self.model.predict(batch_X)

        # Calculate loss
        loss = self.loss(batch_y_hat, batch_y)
        self.log("test_loss", loss)
        return loss

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        batch_X, _ = batch
        return self.model.predict(batch[0])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.nn.functional.l1_loss(y_hat, y, reduction="none"))


class MetricsCallback(L.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.best_test_loss = torch.inf
        self.metric_names = ("train_loss", "test_loss")

    def on_validation_end(self, trainer: L.Trainer, _) -> None:
        metrics = {}
        for metric in self.metric_names:
            if exists := trainer.callback_metrics.get(metric, None):
                metrics[metric] = exists.item()

        # The first test run will be done before training
        if "train_loss" not in metrics:
            tqdm.write(f"Initial (untrained) testing loss: {metrics['test_loss']:.4f}")
            return

        tqdm.write(
            f"Completed {trainer.current_epoch + 1} / {trainer.max_epochs} epochs:"
        )
        if "train_loss" in metrics:
            tqdm.write(f"... mean training loss: {metrics['train_loss']:.4f}")
        if "test_loss" in metrics:
            tqdm.write(f"... mean testing loss: {metrics['test_loss']:.4f}")
            if metrics["test_loss"] < self.best_test_loss:
                tqdm.write(
                    f"... testing loss improvement: {self.best_test_loss - metrics['test_loss']:.4f}"
                )
                self.best_test_loss = metrics["test_loss"]
