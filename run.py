import argparse
import pathlib
import random
import shutil
from contextlib import suppress

import lightning as L
import numpy as np
import torch
import torch.optim as optim
import tqdm
from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
    NUM_FORECAST_STEPS,
)
from cloudcasting.dataset import SatelliteDataset, ValidationSatelliteDataset
from cloudcasting.utils import numpy_validation_collate_fn
from lightning.pytorch.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary

from ocf_iam4vp import IAM4VP, IAM4VPLightning, MetricsCallback


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def summarise(
    batch_size: int,
    device: str,
    hidden_channels_space: int,
    hidden_channels_time: int,
    num_convolutions_space: int,
    num_convolutions_time: int,
    num_forecast_steps: int,
    num_history_steps: int,
) -> None:
    # Create the model
    model = IAM4VP(
        (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
        num_forecast_steps=num_forecast_steps,
        hid_S=hidden_channels_space,
        hid_T=hidden_channels_time,
        N_S=num_convolutions_space,
        N_T=num_convolutions_time,
    )
    model = model.to(device)
    model.train()

    # Create some random inputs
    batch_X = torch.randn(
        batch_size,
        num_history_steps,
        NUM_CHANNELS,
        IMAGE_SIZE_TUPLE[0],
        IMAGE_SIZE_TUPLE[1],
    )
    # Summarise the model
    print(f"- batch-size: {batch_size}")
    print(f"- hidden-channels-space: {hidden_channels_space}")
    print(f"- hidden-channels-time: {hidden_channels_time}")
    print(f"- num-convolutions-space: {num_convolutions_space}")
    print(f"- num-convolutions-time: {num_convolutions_time}")
    print(f"- num-forecast-steps: {num_forecast_steps}")
    print(f"- num-history-steps: {num_history_steps}")
    summary(model, input_data=(batch_X, [], None), device=device)


def train(
    batch_size: int,
    device: str,
    hidden_channels_space: int,
    hidden_channels_time: int,
    num_convolutions_space: int,
    num_convolutions_time: int,
    num_epochs: int,
    num_forecast_steps: int,
    num_history_steps: int,
    output_directory: pathlib.Path,
    training_data_path: str | list[str],
    num_workers: int = 4,
) -> None:
    # Load the training and test datasets
    dataset = SatelliteDataset(
        zarr_path=training_data_path,
        start_time=None,
        end_time=None,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=num_forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )
    print(f"Loaded {len(dataset)} sequences of cloud coverage data.")
    train_length = int(0.9 * len(dataset))
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length)
    )
    print(f"  {len(train_dataset)} will be used for training.")
    print(f"  {len(test_dataset)} will be used for testing")

    # Construct appropriate data loaders
    gen = torch.Generator()
    gen.manual_seed(0)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        generator=gen,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        generator=gen,
    )

    # Create the model
    model = IAM4VPLightning(
        (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
        num_forecast_steps=num_forecast_steps,
        hid_S=hidden_channels_space,
        hid_T=hidden_channels_time,
        N_S=num_convolutions_space,
        N_T=num_convolutions_time,
    )
    model = model.to(device)

    # Log parameters
    print("Training IAM4VP model")
    print(f"... hidden_channels_space {model.hparams['hid_S']}")
    print(f"... hidden_channels_time {model.hparams['hid_T']}")
    print(f"... num_convolutions_space {model.hparams['N_S']}")
    print(f"... num_convolutions_time {model.hparams['N_S']}")
    print(f"... num_forecast_steps {model.hparams['num_forecast_steps']}")
    print(f"... num_history_steps {num_history_steps}")
    print(f"... output_directory {output_directory}")

    # Initialise the trainer
    val_every_n_epochs = 1
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_directory,
        save_top_k=-1,
        every_n_epochs=val_every_n_epochs,
        monitor="test_loss",
        filename="{epoch}-{test_loss:.2f}",
    )
    metrics_callback = MetricsCallback()
    trainer = L.Trainer(
        logger=False,
        callbacks=[checkpoint_callback, metrics_callback],
        max_epochs=num_epochs,
        check_val_every_n_epoch=val_every_n_epochs,
    )

    # Perform training and validation
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )


def validate(
    batch_size: int,
    device: str,
    hidden_channels_space: int,
    hidden_channels_time: int,
    num_convolutions_space: int,
    num_convolutions_time: int,
    num_history_steps: int,
    output_directory: pathlib.Path,
    state_dict_path: str,
    validation_data_path: str | list[str],
    num_workers: int = 0,
) -> None:
    # Create the model
    model = IAM4VP(
        (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
        num_forecast_steps=NUM_FORECAST_STEPS,
        hid_S=hidden_channels_space,
        hid_T=hidden_channels_time,
        N_S=num_convolutions_space,
        N_T=num_convolutions_time,
    )
    model.load_state_dict(
        torch.load(state_dict_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # Log parameters
    print("Validating IAM4VP model")
    print(f"... hidden_channels_space {hidden_channels_space}")
    print(f"... hidden_channels_time {hidden_channels_time}")
    print(f"... num_convolutions_space {num_convolutions_space}")
    print(f"... num_convolutions_time {num_convolutions_time}")
    print(f"... num_forecast_steps {NUM_FORECAST_STEPS}")
    print(f"... num_history_steps {num_history_steps}")
    print(f"... output_directory {output_directory}")

    # Set up the validation dataset
    valid_dataset = ValidationSatelliteDataset(
        zarr_path=validation_data_path,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=NUM_FORECAST_STEPS * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )
    print(f"Loaded {len(valid_dataset)} sequences of cloud coverage data.")

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_validation_collate_fn,
        drop_last=False,
    )

    def plot_channels(y: np.ndarray, y_hat: np.ndarray, name: str) -> None:
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
        fig.savefig(f"{output_directory}/{name}.png")
        plt.close()

    def plot_times(y: np.ndarray, y_hat: np.ndarray, name: str) -> None:
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
        fig.savefig(f"{output_directory}/{name}.png")
        plt.close()

    for idx, (X, y) in enumerate(tqdm.tqdm(valid_dataloader)):
        # Ensure that the ground truth has at least one timestep
        if y.shape[2] < 1:
            continue

        if idx % 100 == 0:
            y_hats = model.predict(torch.from_numpy(X).to(device))
            y_hats_np = y_hats.cpu().detach().numpy()
            del y_hats

            # Plot channels for timestep 1
            y_t1 = y[0, :, 0, :, :]
            y_hat_t1 = y_hats_np[0, :, 0, :, :]
            plot_channels(y_t1, y_hat_t1, name=f"cloud-channels-t1-{idx}")
            del y_t1, y_hat_t1

            # Plot video for channel 6
            y_c6 = y[0, 5, :, :, :]
            y_hat_c6 = y_hats_np[0, 5, :, :, :]
            plot_times(y_c6, y_hat_c6, name=f"cloud-times-c6-{idx}")
            del y_c6, y_hat_c6

            # Plotting/prediction cleanup
            del y_hats_np

        # Iteration cleanup
        del X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--train", action="store_true", help="Run training")
    cmd_group.add_argument(
        "--summarise", action="store_true", help="Print a model summary"
    )
    cmd_group.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=2)
    parser.add_argument("--data-path", type=str, help="Path to the input data")
    parser.add_argument(
        "--hidden-channels-space",
        type=int,
        help="Number of spatial hidden channels",
        default=64,
    )
    parser.add_argument(
        "--hidden-channels-time",
        type=int,
        help="Number of temporal hidden channels",
        default=512,
    )
    parser.add_argument(
        "--num-convolutions-space",
        type=int,
        help="Number of spatial convolutions",
        default=4,
    )
    parser.add_argument(
        "--num-convolutions-time",
        type=int,
        help="Number of temporal convolutions",
        default=6,
    )
    parser.add_argument("--num-epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument(
        "--num-forecast-steps",
        type=int,
        help="Forecast steps used to mitigate error accumulation",
        default=10,
    )
    parser.add_argument(
        "--num-history-steps", type=int, help="History steps", default=24
    )
    parser.add_argument("--model-state-dict", type=str, help="Path to model state dict")
    parser.add_argument("--output-directory", type=str, help="Path to save outputs to")
    args = parser.parse_args()

    # Get the appropriate PyTorch device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Apply constraints on timesteps
    if args.num_forecast_steps > args.num_history_steps:
        msg = f"--num-forecast-steps must be no greater than --num-history-steps for time embedding to work."
        raise ValueError(msg)
    if args.num_history_steps < NUM_FORECAST_STEPS:
        msg = f"--num-history-steps must be at least {NUM_FORECAST_STEPS} for validation to work."
        raise ValueError(msg)

    # Ensure output directory exists
    if args.output_directory:
        output_directory = pathlib.Path(args.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

    if args.train:
        training_data_path = [
            str(path.resolve()) for path in pathlib.Path(args.data_path).glob("*.zarr")
        ]
        train(
            batch_size=args.batch_size,
            device=device,
            hidden_channels_space=args.hidden_channels_space,
            hidden_channels_time=args.hidden_channels_time,
            num_convolutions_space=args.num_convolutions_space,
            num_convolutions_time=args.num_convolutions_time,
            num_epochs=args.num_epochs,
            num_forecast_steps=args.num_forecast_steps,
            num_history_steps=args.num_history_steps,
            output_directory=output_directory,
            training_data_path=training_data_path,
        )
    if args.summarise:
        summarise(
            batch_size=args.batch_size,
            device=device,
            hidden_channels_space=args.hidden_channels_space,
            hidden_channels_time=args.hidden_channels_time,
            num_convolutions_space=args.num_convolutions_space,
            num_convolutions_time=args.num_convolutions_time,
            num_forecast_steps=args.num_forecast_steps,
            num_history_steps=args.num_history_steps,
        )
    if args.validate:
        validation_data_path = [
            str(path.resolve()) for path in pathlib.Path(args.data_path).glob("*.zarr")
        ]
        validate(
            batch_size=args.batch_size,
            device=device,
            hidden_channels_space=args.hidden_channels_space,
            hidden_channels_time=args.hidden_channels_time,
            num_convolutions_space=args.num_convolutions_space,
            num_convolutions_time=args.num_convolutions_time,
            num_history_steps=args.num_history_steps,
            output_directory=output_directory,
            state_dict_path=args.model_state_dict,
            validation_data_path=validation_data_path,
        )
