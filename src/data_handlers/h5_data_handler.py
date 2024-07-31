import torch
from h5py import File
from torchvision import transforms
import numpy as np


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
        self.X = None
        self.y = None
        self.transform = transform
        # Choose either train or test subsets
        self.data_type = "train" if train else "test"
        self.is_rgb = is_rgb
        with File(self.file_path, "r") as file:
            self.dataset_len = len(file[self.data_type]["stimulus"][:500])

    def __getitem__(self, idx: int):
        if self.X is None or self.y is None:
            h5file = File(self.file_path, "r")
            # Read as numpy array
            self.X = np.asarray(h5file[self.data_type]["stimulus"][:500])
            self.y = np.asarray(
                h5file[self.data_type]["response"][self.response_type][:500]
            )
            # Swap axes of y since it is channels last
            self.y = np.transpose(self.y, axes=None)
            self.y = self.y.astype("float32")

        x = self.X[idx]

        if self.is_rgb:
            x = np.repeat(x[..., np.newaxis], 3, -1)

        # transform the data
        x = self.transform(x)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len
