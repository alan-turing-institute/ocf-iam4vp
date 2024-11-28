import argparse
import pathlib

import lightning as L
import torch
import yaml
from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
    NUM_FORECAST_STEPS,
)
from cloudcasting.dataset import SatelliteDataset, ValidationSatelliteDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchinfo import summary

from ocf_iam4vp import (
    IAM4VP,
    EarlyEpochStopping,
    IAM4VPLightning,
    MetricsLogger,
    PlottingCallback,
)


def summarise(
    batch_size: int,
    hidden_channels_space: int,
    hidden_channels_time: int,
    num_convolutions_space: int,
    num_convolutions_time: int,
    num_forecast_steps: int,
    num_history_steps: int,
) -> None:
    # Get the appropriate PyTorch device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

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
        NUM_CHANNELS,
        num_history_steps,
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
    summary(model, input_data=(batch_X, []), device=device)


def train(
    batch_size: int,
    batches_per_checkpoint: int,
    hidden_channels_space: int,
    hidden_channels_time: int,
    num_convolutions_space: int,
    num_convolutions_time: int,
    num_epochs: int,
    num_forecast_steps: int,
    num_history_steps: int,
    output_directory: pathlib.Path,
    resume_from_checkpoint: pathlib.Path | None,
    training_data_path: str | list[str],
    num_workers: int,
) -> None:
    # Set random seeds
    L.seed_everything(42, workers=True)

    # Load the data used to train and test the model
    dataset = SatelliteDataset(
        zarr_path=training_data_path,
        start_time=None,
        end_time=None,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=num_forecast_steps * DATA_INTERVAL_SPACING_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
    )
    n_batches = int(len(dataset) / batch_size)
    print(f"Loaded {len(dataset)} cloud coverage data entries.")
    print(f"... these will be grouped into {n_batches} batches.")

    # Divide into training and test datasets
    val_every_n_batches = (
        batches_per_checkpoint if batches_per_checkpoint > 0 else n_batches
    )
    test_length = min(2 * val_every_n_batches, 0.2 * len(dataset))
    train_length = len(dataset) - test_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length)
    )
    print(
        f"... {int(len(train_dataset) / batch_size)} batches will be used for training."
    )
    print(f"    with testing run after each {val_every_n_batches} batches")
    print(f"... {int(len(test_dataset) / batch_size)} batches will be used for testing")

    # Construct appropriate data loaders
    train_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=test_dataset,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    print("Data loaders will use:")
    worker_type = "persistent" if train_dataloader.num_workers > 0 else "ephemeral"
    print(f"... {train_dataloader.num_workers} {worker_type} workers")
    print(f"... {train_dataloader.batch_size} entries per batch")
    print(f"... {'pinned' if train_dataloader.pin_memory else 'unpinned'} memory")

    # Create the model
    if resume_from_checkpoint:
        print("Loading IAM4VP model from checkpoint")
        model = IAM4VPLightning.load_from_checkpoint(checkpoint_path)
        assert model.hparams["hid_S"] == hidden_channels_space
        assert model.hparams["hid_T"] == hidden_channels_time
        assert model.hparams["N_S"] == num_convolutions_space
        assert model.hparams["N_T"] == num_convolutions_time
        assert model.hparams["num_forecast_steps"] == num_forecast_steps
        assert model.hparams["shape_in"][0] == num_history_steps
    else:
        print("Creating IAM4VP model")
        model = IAM4VPLightning(
            (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
            num_forecast_steps=num_forecast_steps,
            hid_S=hidden_channels_space,
            hid_T=hidden_channels_time,
            N_S=num_convolutions_space,
            N_T=num_convolutions_time,
        )

    # Log parameters
    model.describe({"output_directory": output_directory})

    # Set callbacks
    batch_checkpoint_callback = ModelCheckpoint(
        auto_insert_metric_name=False,
        dirpath=output_directory,
        filename="epoch-{epoch}-batch-{batch_idx:.0f}-loss-{test_loss:.4f}",
        monitor="test_loss",
        mode="min",
        save_on_train_epoch_end=False,  # save after every validation step
        save_top_k=3,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        auto_insert_metric_name=False,
        dirpath=output_directory,
        filename="epoch-{epoch}-loss-{test_loss:.4f}",
        save_on_train_epoch_end=True,  # check after every training epoch
    )
    early_stopping_callback = EarlyEpochStopping(
        check_on_train_epoch_end=False,  # check after every validation step
        min_delta=0.0005,
        monitor="test_loss",
        mode="min",
        patience=3,
    )
    metric_logging_callback = MetricsLogger()

    # List the callback settings
    print("Epoch will terminate early if:")
    print(
        f"... {early_stopping_callback.monitor} improves by less than"
        f" {abs(early_stopping_callback.min_delta)} for"
        f" {early_stopping_callback.patience} consecutive test steps."
    )

    # Initialise the trainer
    trainer = L.Trainer(
        callbacks=[
            metric_logging_callback,
            batch_checkpoint_callback,
            epoch_checkpoint_callback,
            early_stopping_callback,
        ],
        logger=False,
        max_epochs=num_epochs,
        precision="bf16-mixed",
        val_check_interval=val_every_n_batches,
    )

    # Examine the data
    batch_X, _ = next(iter(train_dataloader))
    print("Training data has:")
    print(f"... {batch_X.shape[1]} channels")
    print(f"... {batch_X.shape[3]} pixels (height)")
    print(f"... {batch_X.shape[4]} pixels (width)")

    # Perform training and validation
    print("Start training IAM4VP model")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    print("Finished training IAM4VP model")


def validate(
    batch_size: int,
    checkpoint_path: str,
    num_workers: int,
    output_directory: pathlib.Path,
    validation_data_path: str | list[str],
) -> None:
    # Set random seeds
    L.seed_everything(42, workers=True)

    # Load the pretrained model
    model = IAM4VPLightning.load_from_checkpoint(
        checkpoint_path, num_forecast_steps=NUM_FORECAST_STEPS
    )
    num_history_steps = model.hparams["shape_in"][0]

    # Log parameters
    print("Validating IAM4VP model")
    model.describe({"output_directory": output_directory})

    # Set up the validation dataset
    valid_dataset = ValidationSatelliteDataset(
        forecast_mins=NUM_FORECAST_STEPS * DATA_INTERVAL_SPACING_MINUTES,
        history_mins=(num_history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=True,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        zarr_path=validation_data_path,
    )
    print(f"Loaded {len(valid_dataset)} sequences of cloud coverage data.")
    valid_length = len(valid_dataset)
    print(f"  {valid_length} will be used for validation.")

    valid_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=valid_dataset,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        shuffle=False,
    )

    # Examine the data
    batch_X, _ = next(iter(valid_dataloader))
    print("Validation data has:")
    print(f"... {batch_X.shape[1]} channels")
    print(f"... {batch_X.shape[3]} pixels (height)")
    print(f"... {batch_X.shape[4]} pixels (width)")

    # Initialise the predictor and plot outputs
    plotting_callback = PlottingCallback(
        every_n_batches=3, output_directory=output_directory
    )
    predictor = L.Trainer(callbacks=[plotting_callback], logger=False)
    predictor.predict(model, valid_dataloader)
    print("Finished validating IAM4VP model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--train", action="store_true", help="Run training")
    cmd_group.add_argument(
        "--summarise", action="store_true", help="Print a model summary"
    )
    cmd_group.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=2)
    parser.add_argument(
        "--batches-per-checkpoint",
        type=int,
        help="How many batches per checkpoint",
        default=-1,
    )
    parser.add_argument("--data-path", type=str, help="Path to the input data")
    parser.add_argument(
        "--hidden-channels-space",
        type=int,
        help="Number of spatial hidden channels",
        default=32,
    )
    parser.add_argument(
        "--hidden-channels-time",
        type=int,
        help="Number of temporal hidden channels",
        default=64,
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
        default=4,
    )
    parser.add_argument("--num-epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument(
        "--num-forecast-steps",
        type=int,
        help="Forecast steps used to mitigate error accumulation",
        default=12,
    )
    parser.add_argument(
        "--num-history-steps", type=int, help="History steps", default=24
    )
    parser.add_argument(
        "--num-workers", type=int, help="Number of workers to use", default=0
    )
    parser.add_argument("--output-directory", type=str, help="Path to save outputs to")
    parser.add_argument(
        "--resume-from-checkpoint", type=str, help="Path to a checkpoint file"
    )
    parser.add_argument(
        "--validate-config-file", type=str, help="Validation config file"
    )
    args = parser.parse_args()

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
        checkpoint_path = (
            pathlib.Path(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
        train(
            batch_size=args.batch_size,
            batches_per_checkpoint=args.batches_per_checkpoint,
            hidden_channels_space=args.hidden_channels_space,
            hidden_channels_time=args.hidden_channels_time,
            num_convolutions_space=args.num_convolutions_space,
            num_convolutions_time=args.num_convolutions_time,
            num_epochs=args.num_epochs,
            num_forecast_steps=args.num_forecast_steps,
            num_history_steps=args.num_history_steps,
            num_workers=args.num_workers,
            output_directory=output_directory,
            resume_from_checkpoint=checkpoint_path,
            training_data_path=training_data_path,
        )
    if args.summarise:
        summarise(
            batch_size=args.batch_size,
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
        with open(args.validate_config_file, "r") as f_config:
            config = yaml.safe_load(f_config)
        validate(
            batch_size=args.batch_size,
            checkpoint_path=config["model"]["params"]["checkpoint_path"],
            num_workers=args.num_workers,
            output_directory=output_directory,
            validation_data_path=validation_data_path,
        )
