import torch
from h5py import File
from torchvision import transforms
import numpy as np
from numpy import ndarray, dtype
from typing import Any


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        response_type: str,
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        train: bool = True,
        is_rgb: bool = False,
    ):
        self.file_path = path
        # The available types are firing_rate_10ms, binned
        self.response_type = response_type
        self.X: ndarray[Any, dtype[Any]] | None = None
        self.Y: ndarray[Any, dtype[Any]] | None = None
        self.transform = transform
        # Choose either train or test subsets
        self.data_type = "train" if train else "test"
        self.is_rgb = is_rgb
        with File(self.file_path, "r") as file:
            self.dataset_len = len(file[self.data_type]["stimulus"][:500])

    def __getitem__(self, idx: int):
        if self.X is None or self.Y is None:
            h5file = File(self.file_path, "r")
            # Read as numpy array
            X = np.asarray(h5file[self.data_type]["stimulus"][:500])
            Y = np.asarray(h5file[self.data_type]["response"][self.response_type][:500])
            # Swap axes of y since it is channels last
            Y = np.transpose(Y, axes=None)
            Y = Y.astype("float32")
            self.X = X
            self.Y = Y

        x = self.X[idx]

        if self.is_rgb:
            x = np.repeat(x[..., np.newaxis], 3, -1)

        # transform the data
        x = self.transform(x)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len
