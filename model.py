from cloudcasting.constants import (
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
    NUM_FORECAST_STEPS,
)
from cloudcasting.models import AbstractModel

import torch
import numpy as np
from ocf_iam4vp import IAM4VP


# We define a new class that inherits from AbstractModel
class IAM4VPCloudcaster(AbstractModel):
    """IAM4VPCloudcaster model class"""

    def __init__(
        self,
        hidden_channels_space: int,
        hidden_channels_time: int,
        num_convolutions_space: int,
        num_convolutions_time: int,
        num_forecast_steps: int,
        num_history_steps: int,
        num_training_epochs: int,
        state_dict_path: str,
        version: str,
    ) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data.
        # The history_steps parameter should be specified in `validate_config.yml`, along with
        # any other parameters (replace `example_parameter` with as many other parameters as you need to initialize your model, and also add them to `validate_config.yml` under `model: params`)
        super().__init__(num_history_steps)

        self.hyperparameters = {
            "hidden_channels_space": hidden_channels_space,
            "hidden_channels_time": hidden_channels_time,
            "num_convolutions_space": num_convolutions_space,
            "num_convolutions_time": num_convolutions_time,
            "num_forecast_steps": num_forecast_steps,
            "num_history_steps": num_history_steps,
            "num_training_epochs": num_training_epochs,
            "version": version,
        }

        # Get the appropriate PyTorch device
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load the pretrained model
        self.model = IAM4VP(
            (num_history_steps, NUM_CHANNELS, IMAGE_SIZE_TUPLE[0], IMAGE_SIZE_TUPLE[1]),
            hid_S=hidden_channels_space,
            hid_T=hidden_channels_time,
            N_S=num_convolutions_space,
            N_T=num_convolutions_time,
        )
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=self.device, weights_only=True)
        )

        # Move to the appropriate device and set evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, X: np.ndarray) -> np.ndarray:
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)
        # The output y_hat is a numpy array with shape (batch_size, channels, time, height, width)
        return self.model.predict(X, NUM_FORECAST_STEPS, self.device)

    def hyperparameters_dict(self) -> dict[str, str]:
        """
        Record hyperparameters

        This function returns a dictionary of hyperparameters for the model that will
        be saved with the model scores to wandb.
        """
        return self.hyperparameters
