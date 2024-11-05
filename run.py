import argparse
import pathlib
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
)
from cloudcasting.dataset import SatelliteDataset, ValidationSatelliteDataset
from cloudcasting.utils import numpy_validation_collate_fn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary

from ocf_iam4vp import IAM4VP


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
    num_history_steps: int,
) -> None:
    # Create the model
    model = IAM4VP(
        (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
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
    times = torch.tensor(100).repeat(batch_X.shape[0]).to(device)

    # Summarise the model
    print(f"- batch-size: {batch_size}")
    print(f"- hidden-channels-space: {hidden_channels_space}")
    print(f"- hidden-channels-time: {hidden_channels_time}")
    print(f"- num-convolutions-space: {num_convolutions_space}")
    print(f"- num-convolutions-time: {num_convolutions_time}")
    print(f"- num-history-steps: {num_history_steps}")
    summary(model, input_data=(batch_X, [], times), device=device)


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
    num_workers: int = 0,
) -> None:
    # Load the training dataset
    dataset = SatelliteDataset(
        zarr_path=training_data_path,
        start_time="2022-01-31",
        end_time=None,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=num_forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )

    # Construct a DataLoader
    gen = torch.Generator()
    gen.manual_seed(0)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    # Create the model
    model = IAM4VP(
        (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
        hid_S=hidden_channels_space,
        hid_T=hidden_channels_time,
        N_S=num_convolutions_space,
        N_T=num_convolutions_time,
    )
    model = model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    best_loss = 999
    best_model = None
    save_model = False
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        for batch_X, batch_y in tqdm.tqdm(train_dataloader):
            # Send batch tensors to the current device
            batch_X = batch_X.swapaxes(1, 2).to(device)
            batch_y = batch_y.swapaxes(1, 2).to(device)

            y_hats = []
            for f_step in range(num_forecast_steps):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Generate an appropriately-sized set of blank times
                times = torch.tensor(f_step * 100).repeat(batch_X.shape[0]).to(device)

                # Forward pass for the next time step
                y_hat = model(batch_X, times, y_hats)

                # Calculate the loss
                loss = criterion(y_hat, batch_y[:, f_step, :, :, :])

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update best model so-far if appropriate
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model = model
                    save_model = True

                # Append latest prediction to queue
                y_hats.append(y_hat.detach())

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best loss {best_loss:.4f}"
        )
        if save_model:
            torch.save(
                best_model.state_dict(),
                output_directory
                / f"best-model-epoch-{epoch}-loss-{best_loss:.3g}.state-dict.pt",
            )
            save_model = False

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
        hid_S=hidden_channels_space,
        hid_T=hidden_channels_time,
        N_S=num_convolutions_space,
        N_T=num_convolutions_time,
    )
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Set up the validation dataset
    num_forecast_steps = 1
    valid_dataset = ValidationSatelliteDataset(
        zarr_path=validation_data_path,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=num_forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_validation_collate_fn,
        drop_last=False,
    )

    def plot(y: torch.Tensor, y_hat: torch.Tensor, name: str) -> None:
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(y, cmap="gray")
        ax2.imshow(y_hat, cmap="gray")
        fig.savefig(f"{output_directory}/{name}.png")
        plt.close()

    times = torch.tensor(100).repeat(batch_size).to(device)
    for idx, (X, y) in enumerate(tqdm.tqdm(valid_dataloader)):
        if idx % 100 == 0:
            X = torch.from_numpy(X).swapaxes(1, 2).to(device)
            y = torch.from_numpy(y).swapaxes(1, 2).to(device)
            y_hat = model(X, times)
            plot(y[0][0][0], y_hat[0][0], name=f"cloud-{idx}")

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
