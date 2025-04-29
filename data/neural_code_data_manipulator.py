import h5py
import sys
from pathlib import Path
from logging import Logger
import numpy as np


class NeuralCodeData:
    """
    A class to handle neural code data and create a validation split.
    Attributes:
    -----------
    path : Path
        The file path to the data.
    logger : Logger
        The logger instance for logging information and errors.
    Methods:
    --------
    read_data():
        Reads the data from the specified path and handles any errors that occur during reading.
    make_validation_split(train_ratio: float):
        Creates a validation split from the training data and saves the split data to a new file.
        Parameters:
        -----------
        train_ratio : float
            The ratio of the data to be used for training. The rest will be used for validation.
    """  # noqa: E501

    def __init__(self, path: Path, logger: Logger):
        self.path = path
        self.logger = logger

    def read_data(self):
        try:
            self.data = h5py.File(self.path, "r")
        except Exception as e:
            self.logger.error(f"Error reading data from {self.path}")
            self.logger.error(e)
            sys.exit(1)

    def make_validation_split(self, train_ratio: float):
        # Create a new path for the file with the validation split
        new_path = self.path.parent / f"{self.path.stem}_with_val.h5"
        # Get keys from the data train response
        data_keys = list(self.data["train"]["response"].keys())
        with h5py.File(new_path, "w") as f:
            # Copy stimulus data
            train_len = int(self.data["train"]["stimulus"].shape[0] * train_ratio)
            val_len = self.data["train"]["stimulus"].shape[0] - train_len
            self.logger.info(f"Train length: {train_len} Validation length: {val_len}")
            f.create_dataset(
                "train/stimulus", data=self.data["train"]["stimulus"][:train_len]
            )
            f.create_dataset(
                "val/stimulus", data=self.data["train"]["stimulus"][train_len:]
            )
            f.create_dataset("test/stimulus", data=self.data["test"]["stimulus"])
            for key in data_keys:
                # Calculate the length of the training data
                train_matrix = self.data["train"]["response"][key]
                self.logger.info(
                    f"Original train data {key} shape {train_matrix.shape}"
                )
                self.logger.info(f"Splitting {key}")
                # Split the data
                train_data = train_matrix[..., :train_len]
                val_data = train_matrix[..., train_len:]
                test_data = self.data["test"]["response"][key]
                # Save the data
                f.create_dataset(f"train/response/{key}", data=train_data)
                f.create_dataset(f"val/response/{key}", data=val_data)
                f.create_dataset(f"test/response/{key}", data=test_data)

    def add_gaussian_noise(self, key:str="test", sigma:float=0.3):
        """
        Adds Gaussian noise to stimulus.
        Parameters:         
        -----------
        noise_level : float
            The standard deviation of the Gaussian noise to be added.
        """

        # Create a new path for the file with noised data
        new_path = self.path.parent / f"{self.path.stem}_{key}_noised.h5"

        images = self.data[key]["stimulus"]
        noised = images + np.random.normal(0, sigma, images.shape)

        with h5py.File(new_path, "w") as f:
            f.create_dataset(
                f"{key}/stimulus", data=noised
            )
            # Save response data
            data_keys = list(self.data[key]["response"].keys())
            for k in data_keys:
                # Get the response data
                response_data = self.data[key]["response"][k]
                # Save the response data
                f.create_dataset(f"{key}/response/{k}", data=response_data)

    def close(self):
        """
        Closes the data file.
        """
        self.data.close()
        self.logger.info(f"Closed data file {self.path}")