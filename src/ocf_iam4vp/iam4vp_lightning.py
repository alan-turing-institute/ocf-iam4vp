import time
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

from .iam4vp import IAM4VP

LightningBatch = tuple[torch.Tensor, torch.Tensor]


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
        num_channels: int,
        num_history_steps: int,
        num_forecast_steps: int = 12,
        hid_S: int = 64,
        hid_T: int = 512,
        N_S: int = 4,
        N_T: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = IAM4VP(
            num_channels, num_history_steps, num_forecast_steps, hid_S, hid_T, N_S, N_T
        )
        # Enable manual optimisation to reduce memory usage in the forecast loop
        # This means that we have make the backward pass and optimizer calls explicit
        self.automatic_optimization = False
        # Reduce precision to take advantage of hardware optimisations
        torch.set_float32_matmul_precision("medium")
        # Set this attribute to stop the epoch early
        self.stop_epoch = False

    def describe(self, extra_values: dict[str, str] = {}) -> None:
        print(f"... hidden_channels_space {self.hparams['hid_S']}")
        print(f"... hidden_channels_time {self.hparams['hid_T']}")
        print(f"... num_convolutions_space {self.hparams['N_S']}")
        print(f"... num_convolutions_time {self.hparams['N_T']}")
        print(f"... num_forecast_steps {self.hparams['num_forecast_steps']}")
        print(f"... num_history_steps {self.hparams['num_history_steps']}")
        for key, value in extra_values.items():
            print(f"... {key} {value}")

    def on_train_batch_start(self, batch: LightningBatch, batch_idx: int) -> int | None:
        if self.stop_epoch:
            self.stop_epoch = False
            return -1

    def training_step(self, batch: LightningBatch, batch_idx: int) -> torch.Tensor:
        # Split the batch into X and y
        self.log("batch_idx", batch_idx + 1)  # counting starts from 0
        batch_X, batch_y = batch

        # Get optimizers
        optimizers = self.optimizers()

        # Prepare prediction and loss lists
        y_hats: list[torch.Tensor] = []
        losses: list[float] = []

        # Generate the requested number of forecasts
        for idx_forecast in range(self.model.num_forecast_steps):
            # Zero the parameter gradients
            optimizers.zero_grad()

            # Forward pass for the next time step (batch_size, channels, height, width)
            y_hat = self.model(batch_X, y_hats)

            # Calculate the loss
            loss = self.loss(y_hat, batch_y[:, :, idx_forecast, :, :])

            # Backward pass and optimize
            self.manual_backward(loss)
            optimizers.step()

            # Keep track of loss values
            losses.append(loss.detach())
            del loss

            # Detach latest prediction to save memory before adding it to the queue
            y_hats.append(y_hat.detach())

        # Free up memory
        del batch_X, batch_y

        # Calculate mean loss for the full set of forecasts
        loss = torch.stack(losses).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: LightningBatch, batch_idx: int) -> torch.Tensor:
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
        batch: LightningBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.predict(batch[0])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.AdamW(self.model.parameters(), lr=1e-4)

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha_gdl: int = 2,
        lambda_gdl: int = 10,
        lambda_mae: int = 1,
    ) -> torch.Tensor:
        # Mean absolute error
        loss_mae = torch.nanmean(
            torch.nn.functional.l1_loss(y_hat, y, reduction="none")
        )

        # Gradient difference loss from https://arxiv.org/abs/1511.05440
        loss_gdl = torch.nanmean(
            (y_hat.diff(axis=-2).abs_() - y.diff(axis=-2).abs_()).pow(alpha_gdl)
        ) + torch.nanmean(
            (y_hat.diff(axis=-1).abs_() - y.diff(axis=-1).abs_()).pow(alpha_gdl)
        )

        # Return combination of two losses
        return lambda_gdl * loss_gdl + lambda_mae * loss_mae


class EarlyEpochStopping(EarlyStopping):
    """PyTorch Lightning epoch stopping callback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reason = ""
        self.stop_epoch = False

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: LightningBatch,
        batch_idx: int,
    ) -> None:
        if self.stop_epoch:
            tqdm.write(f"Stopping epoch {trainer.current_epoch} early. {self.reason}")
            pl_module.stop_epoch = True

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.best_score = (
            torch.tensor(torch.inf)
            if self.monitor_op == torch.lt
            else -torch.tensor(torch.inf)
        )
        self.stop_epoch = False
        self.wait_count = 0

    def _run_early_stopping_check(self, trainer: L.Trainer) -> None:
        logs = trainer.callback_metrics
        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        self.stop_epoch = self.stop_epoch or should_stop
        self.reason = (
            (reason if reason else "").replace("Signaling Trainer to stop.", "").strip()
        )


class MetricsLogger(L.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self) -> None:
        super().__init__()
        self.best_test_loss = torch.inf
        self.metric_names = ("train_loss", "test_loss")
        self.start_time = time.perf_counter()
        self.n_batches_this_epoch = 0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.n_batches_this_epoch = 0

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: LightningBatch,
        batch_idx: int,
    ) -> None:
        self.n_batches_this_epoch += 1

    def on_validation_end(self, trainer: L.Trainer, _) -> None:
        # Reset the timer
        elapsed = time.perf_counter() - self.start_time
        self.start_time = time.perf_counter()

        # Get values for existing metrics
        metrics = {}
        for metric in self.metric_names:
            if exists := trainer.callback_metrics.get(metric, None):
                metrics[metric] = exists.item()

        # The first test run will be done before training
        if "train_loss" not in metrics:
            tqdm.write(f"... mean (untrained) testing loss: {metrics['test_loss']:.4f}")
            self.best_test_loss = metrics["test_loss"]
            return

        rate = trainer.val_check_interval / elapsed
        tqdm.write(
            f"Epoch {trainer.current_epoch}: "
            f"Processed {self.n_batches_this_epoch} batches "
            f"in {tqdm.format_interval(elapsed)} [{rate:.3f}it/s]"
        )
        if "train_loss" in metrics:
            tqdm.write(f"... mean training loss: {metrics['train_loss']:.4f}")
        if "test_loss" in metrics:
            tqdm.write(f"... mean testing loss: {metrics['test_loss']:.4f}")
            if metrics["test_loss"] < self.best_test_loss:
                tqdm.write("... ✨ best testing loss so far ✨")
                self.best_test_loss = metrics["test_loss"]


class PlottingCallback(L.Callback):
    def __init__(self, output_directory: str, every_n_batches: int = 1):
        super().__init__()
        self.output_directory = output_directory
        self.every_n_batches = every_n_batches

    def plot_channels(self, y: np.ndarray, y_hat: np.ndarray, filename: str) -> None:
        """Plot comparison across channels for inputs with shape (C, H, W)"""
        assert y.shape == y_hat.shape
        fig, axs = plt.subplots(3, 2)
        for ix, iy in np.ndindex(axs.shape):
            axs[ix, iy].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
        axs[0, 0].set_ylabel("Channel 1")
        axs[1, 0].set_ylabel("Channel 6")
        axs[2, 0].set_ylabel("Channel 11")
        axs[2, 0].set_xlabel("Ground truth")
        axs[2, 1].set_xlabel("Prediction")
        axs[0, 0].imshow(y[0], cmap="gray")
        axs[1, 0].imshow(y[5], cmap="gray")
        axs[2, 0].imshow(y[10], cmap="gray")
        axs[0, 1].imshow(y_hat[0], cmap="gray")
        axs[1, 1].imshow(y_hat[5], cmap="gray")
        axs[2, 1].imshow(y_hat[10], cmap="gray")
        fig.savefig(f"{self.output_directory}/{filename}.png")
        plt.close()

    def plot_times(self, y: np.ndarray, y_hat: np.ndarray, filename: str) -> None:
        """Plot comparison across times for inputs with shape (T, H, W)"""
        assert y.shape == y_hat.shape
        fig, axs = plt.subplots(3, 2)
        for ix, iy in np.ndindex(axs.shape):
            axs[ix, iy].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
        axs[0, 0].set_ylabel("Time 1")
        axs[1, 0].set_ylabel("Time 6")
        axs[2, 0].set_ylabel("Time 12")
        axs[2, 0].set_xlabel("Ground truth")
        axs[2, 1].set_xlabel("Prediction")
        axs[0, 0].imshow(y[0], cmap="gray")
        axs[1, 0].imshow(y[5], cmap="gray")
        axs[2, 0].imshow(y[11], cmap="gray")
        axs[0, 1].imshow(y_hat[0], cmap="gray")
        axs[1, 1].imshow(y_hat[5], cmap="gray")
        axs[2, 1].imshow(y_hat[11], cmap="gray")
        fig.savefig(f"{self.output_directory}/{filename}.png")
        plt.close()

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        predicted_outputs: torch.Tensor,
        batch: LightningBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""
        if batch_idx % self.every_n_batches == 0:
            tqdm.write(f"Plotting outputs for batch {batch_idx}")
            y = batch[1].cpu().detach().numpy()
            y_hat = predicted_outputs.cpu().detach().numpy()

            # Plot channels for timestep 1
            self.plot_channels(
                y[0, :, 0, :, :],
                y_hat[0, :, 0, :, :],
                filename=f"cloud-channels-t1-{batch_idx}",
            )

            # Plot timesteps for channel 6
            self.plot_times(
                y[0, 5, :, :, :],
                y_hat[0, 5, :, :, :],
                filename=f"cloud-times-c6-{batch_idx}",
            )
