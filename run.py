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
from cloudcasting.dataset import SatelliteDataset
from torch.utils.data import DataLoader

from ocf_iam4vp import IAM4VP


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    batch_size: int,
    device: str,
    forecast_steps: int,
    history_steps: int,
    latent_space_channels: int,
    latent_space_factor: int,
    num_epochs: int,
    output_directory: pathlib.Path,
    training_data_path: str | list[str],
    num_workers: int = 0,
) -> None:
    # Load the training dataset
    dataset = SatelliteDataset(
        zarr_path=training_data_path,
        start_time="2022-01-31",
        end_time=None,
        history_mins=(history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
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
        (history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
        C_latent=latent_space_channels,
        N_S=latent_space_factor,
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
            for f_step in range(forecast_steps):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Generate an appropriately-sized set of blank times
                times = torch.tensor(f_step * 100).repeat(batch_X.shape[0]).to(device)

                # Forward pass for the next time step
                y_hat = model(batch_X, y_hats, times)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--train", action="store_true", help="Run training")
    cmd_group.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=2)
    parser.add_argument("--data-path", type=str, help="Path to the input data")
    parser.add_argument(
        "--latent-space-channels", type=int, help="Number of latent space channels", default=64
    )
    parser.add_argument(
        "--latent-space-factor", type=int, help="Factor to reduce by when transforming to latent space", default=4
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
    output_directory = pathlib.Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.train:
        training_data_path = [
            str(path.resolve()) for path in pathlib.Path(args.data_path).glob("*.zarr")
        ]
        train(
            batch_size=args.batch_size,
            device=device,
            forecast_steps=args.num_forecast_steps,
            history_steps=args.num_history_steps,
            latent_space_channels=args.latent_space_channels,
            latent_space_factor=args.latent_space_factor,
            num_epochs=args.num_epochs,
            output_directory=output_directory,
            training_data_path=training_data_path,
        )
    if args.validate:
        print("Validation is not currently supported")
