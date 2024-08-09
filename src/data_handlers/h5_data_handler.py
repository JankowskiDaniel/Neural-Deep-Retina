import torch
from h5py import File
from torchvision import transforms
import numpy as np
from numpy import ndarray, dtype
from typing import Any, Tuple
from pathlib import Path

transform_x: transforms.Compose = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        response_type: str,
        is_train: bool = True,
        is_rgb: bool = False,
        y_scaler: Any = None,
    ):
        """
        Initializes the H5Dataset object.

        Args:
            path (Path): The path to the .h5 file.
            response_type (str): The type of response data. Available types are 'firing_rate_10ms' and 'binned'.
            is_train (bool, optional): Whether the data is for training or testing. Defaults to True.
            is_rgb (bool, optional): Whether the data is in RGB format. Defaults to False.
            y_scaler (Any, optional): The scaler for the response data. Any scaler from sklearn.preprocessing, for example, StandardScaler. Defaults to None.
        """  # noqa: E501

        self.file_path = path
        # The available types are firing_rate_10ms, binned
        self.response_type = response_type
        # Choose either train or test subsets
        self.data_type = "train" if is_train else "test"
        self.is_train = is_train
        self.is_rgb = is_rgb
        self.y_scaler = y_scaler
        self.transform_x = transform_x
        # Read dataset from file
        X, y = self.read_h5_to_numpy()
        self.dataset_len = len(X)
        self.X: ndarray[Any, dtype[Any]] = X
        self.Y: ndarray[Any, dtype[Any]] = y
        self.input_shape: tuple = X.shape
        self.output_shape: tuple = y.shape

    def read_h5_to_numpy(
        self,
    ) -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Reads data from an HDF5 file and converts it to numpy arrays. Normalizes the output data if the scaler is provided.
        Returns:
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]: A tuple containing the input data (X) and the output data (y).
        """  # noqa: E501
        with File(self.file_path, "r") as h5file:
            # Read as numpy arrays
            X = np.asarray(h5file[self.data_type]["stimulus"][:500])
            y = np.asarray(h5file[self.data_type]["response"][self.response_type])

        y = y.astype("float32")

        # Normalize the output data
        if self.y_scaler is not None:
            y = self.transform_y(y)

        return X, y

    def transform_y(self, y: ndarray[Any, dtype[Any]]) -> ndarray[Any, dtype[Any]]:
        """
        Transforms the target variable 'y' using a scaler.

        Parameters:
        - y: ndarray[Any, dtype[Any]]
            The target variable to be transformed.

        Returns:
        - ndarray[Any, dtype[Any]]
            The transformed target variable.
        """
        if self.is_train:
            # Fit the scaler on the training data and transform the data
            y = self.y_scaler.fit_transform(y)
        else:
            # Only transform the test data
            y = self.y_scaler.transform(y)
        return y

    def __getitem__(self, idx: int):

        x = self.X[idx]
        if self.is_rgb:
            x = np.repeat(x[..., np.newaxis], 3, -1)

        # Transform the image to tensor
        x = self.transform_x(x)
        # Transform the output value to tensor
        y = torch.tensor(self.Y[:, idx], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len
