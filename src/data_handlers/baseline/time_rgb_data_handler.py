import torch
from h5py import File
from torchvision.transforms import v2
import numpy as np
from numpy import ndarray, dtype
from typing import Any, Tuple
from pathlib import Path
import pickle

transform_x: v2.Compose = v2.Compose([v2.ToDtype(torch.float32, scale=True)])


class BaselineRGBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        response_type: str,
        results_dir: Path,
        is_train: bool = True,
        y_scaler: Any = None,
        use_saved_scaler: bool = False,
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
        self.y_scaler = y_scaler
        self.results_dir = results_dir
        self.transform_x = transform_x
        # Allows to use the saved scaler for the train data
        self.use_saved_scaler = use_saved_scaler
        # Read dataset from file
        X, y = self.read_h5_to_numpy()
        self.X: ndarray[Any, dtype[Any]] = X
        self.Y: ndarray[Any, dtype[Any]] = y
        self.input_shape: tuple = X.shape
        self.output_shape: tuple = y.shape

        self.subseq_length: int = 3
        self.dataset_len: int = len(X) - self.subseq_length

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
        if self.y_scaler is not None or self.use_saved_scaler:
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
        y_tran = y.T  # scale the data to the (n_samples, n_features) shape
        if self.is_train and not self.use_saved_scaler:
            # Fit the scaler on the training data and transform the data
            y_fit = self.y_scaler.fit_transform(y_tran)
            # Save the scaler
            with open(self.results_dir / "y_scaler.pkl", "wb") as f:
                pickle.dump(self.y_scaler, f)
        else:
            try:
                # load the scaler
                with open(self.results_dir / "y_scaler.pkl", "rb") as f:
                    self.y_scaler = pickle.load(f)
                # Only transform the test data
                y_fit = self.y_scaler.transform(y_tran)
            except FileNotFoundError:
                print("The scaler file is not found. Target will not be scaled")
                y_fit = y_tran
        y = y_fit.T  # return the data to the original shape
        return y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stack three consecutive grayscale images
        images = []
        for i in range(self.subseq_length):
            x = self.X[idx + i]
            x = torch.from_numpy(x)
            images.append(x)

        # Stack images along the channel dimension
        x = torch.stack(images, dim=0)  # Shape will be (3, H, W)

        # Apply any transformations to the stacked images
        x = self.transform_x(x)

        # Get the target for the fourth image
        y = torch.tensor(self.Y[:, idx + self.subseq_length - 1], dtype=torch.float32)

        return x, y

    def __len__(self):
        return self.dataset_len
