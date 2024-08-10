import torch
from h5py import File
from torchvision.transforms import v2
import numpy as np
from numpy import ndarray, dtype
from typing import Any
from pathlib import Path

transform_x: v2.Compose = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.ElasticTransform(alpha=50.0),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Path,
        is_train: bool = True,
        is_rgb: bool = False,
    ):
        """
        Initializes the H5Dataset object.
        The version for the autoencoder reads only with images.

        Args:
            path (Path): The path to the .h5 file.
            is_train (bool, optional): Whether the data is for training or testing. Defaults to True.
            is_rgb (bool, optional): Whether the data is in RGB format. Defaults to False.
        """  # noqa: E501

        self.file_path = path
        # Choose either train or test subsets
        self.data_type = "train" if is_train else "test"
        self.is_train = is_train
        self.is_rgb = is_rgb
        self.transform_x = transform_x
        # Read dataset from file
        X = self.read_h5_to_numpy()
        self.dataset_len = len(X)
        self.X: ndarray[Any, dtype[Any]] = X
        self.input_shape: tuple = X.shape

    def read_h5_to_numpy(
        self,
        subset_size: int = 2000,
    ) -> ndarray[Any, dtype[Any]]:
        """
        Reads data from an HDF5 file and converts it to numpy arrays.
        Returns:
            ndarray[Any, dtype[Any]]: Images, input data (X).
        """  # noqa: E501
        with File(self.file_path, "r") as h5file:
            # Read as numpy array
            X = np.asarray(h5file[self.data_type]["stimulus"][:subset_size])

        return X

    def __getitem__(self, idx: int):

        x = self.X[idx]
        if self.is_rgb:
            x = np.repeat(x[..., np.newaxis], 3, -1)

        # Apply image transfromations
        x = self.transform_x(x)
        return x

    def __len__(self):
        return self.dataset_len
