import torch
from h5py import File
from torchvision import transforms
import numpy as np
from numpy import ndarray, dtype
from typing import Any, Tuple

transform: transforms.Compose = (
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        response_type: str,
        train: bool = True,
        is_rgb: bool = False,
    ):
        self.file_path = path
        # The available types are firing_rate_10ms, binned
        self.response_type = response_type
        # Choose either train or test subsets
        self.data_type = "train" if train else "test"
        self.is_rgb = is_rgb
        self.transform = transform
        # Read dataset from file
        X, y = self.read_h5_to_numpy()
        self.X: ndarray[Any, dtype[Any]] = X
        self.Y: ndarray[Any, dtype[Any]] = y
        self.input_shape: tuple = X.shape
        self.output_shape: tuple = y.shape

    def read_h5_to_numpy(
        self,
    ) -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        with File(self.file_path, "r") as h5file:
            # Read as numpy arrays
            X = np.asarray(h5file[self.data_type]["stimulus"][:500])
            y = np.asarray(h5file[self.data_type]["response"][self.response_type][:500])
        y = y.astype("float32")

        return X, y

    def __getitem__(self, idx: int):
        if self.X is None or self.Y is None:
            self.read_h5_to_numpy()

        x = self.X[idx]
        if self.is_rgb:
            x = np.repeat(x[..., np.newaxis], 3, -1)

        # Transform the image to tensor
        x = self.transform(x)
        # Transform the output value to tensor
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len
