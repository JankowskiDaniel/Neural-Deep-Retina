import argparse
import h5py
from pathlib import Path
from logging import Logger
import sys

sys.path.append("src")
from src.utils.logger import get_logger  # noqa: E402


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
        # Close the file
        self.data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("-r", "--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    logger = get_logger()

    data = NeuralCodeData(args.path, logger)
    logger.info(f"Reading data from {args.path}")
    data.read_data()
    logger.info(f"Creating a validation split with ratio {args.train_ratio}")
    data.make_validation_split(args.train_ratio)
