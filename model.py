import re

import numpy as np
import torch
from cloudcasting.constants import NUM_FORECAST_STEPS
from cloudcasting.models import AbstractModel

from ocf_iam4vp import IAM4VPLightning


# We define a new class that inherits from AbstractModel
class IAM4VPCloudcaster(AbstractModel):
    """IAM4VPCloudcaster model class"""

    def __init__(
        self,
        checkpoint_path: str,
        version: str,
    ) -> None:
        # Load the pretrained model
        self.model = IAM4VPLightning.load_from_checkpoint(
            checkpoint_path, num_forecast_steps=NUM_FORECAST_STEPS
        )
        num_history_steps = self.model.hparams["shape_in"][0]
        num_training_epochs = int(
            re.search(r"epoch=(\d+)-test_loss.*", checkpoint_path).group(1)
        )

        # All models must include `history_steps` as a parameter. This is the number of
        # previous frames that the model uses to makes its predictions. This should not
        # be more than 25, i.e. 6 hours (inclusive of end points) of 15 minutely data.
        super().__init__(num_history_steps)

        self.hyperparameters = {
            "hidden_channels_space": self.model.hparams["hid_S"],
            "hidden_channels_time": self.model.hparams["hid_T"],
            "num_convolutions_space": self.model.hparams["N_S"],
            "num_convolutions_time": self.model.hparams["N_S"],
            "num_forecast_steps": self.model.hparams["num_forecast_steps"],
            "num_history_steps": num_history_steps,
            "num_training_epochs": num_training_epochs,
            "version": version,
        }

        # Move to the appropriate device and set evaluation mode
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, X: np.ndarray) -> np.ndarray:
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)
        # The output y_hat is a numpy array with shape (batch_size, channels, time, height, width)
        return (
            self.model.model.predict(torch.from_numpy(X).to(self.device))
            .cpu()
            .detach()
            .numpy()
        )

    def hyperparameters_dict(self) -> dict[str, str]:
        """
        Record hyperparameters

        This function returns a dictionary of hyperparameters for the model that will
        be saved with the model scores to wandb.
        """
        return self.hyperparameters
