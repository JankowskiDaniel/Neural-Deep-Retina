import torch
from typing import Any, Tuple
from pathlib import Path
from yaml import safe_load

from h5_data_handler import H5Dataset


class H5SeqDataset(H5Dataset):
    def __init__(
        self,
        path: Path,
        response_type: str,
        is_train: bool = True,
        is_rgb: bool = False,
        y_scaler: Any = None,
        seq_length: int = 10,
    ):
        """
        Initializes the H5Dataset object.

        Args:
            path (Path): The path to the .h5 file.
            response_type (str): The type of response data. Available types are 'firing_rate_10ms' and 'binned'.
            is_train (bool, optional): Whether the data is for training or testing. Defaults to True.
            is_rgb (bool, optional): Whether the data is in RGB format. Defaults to False.
            y_scaler (Any, optional): The scaler for the response data. Any scaler from sklearn.preprocessing, for example, StandardScaler. Defaults to None.
            seq_length (int, optional): The length of the sequence. Defaults to 10.
        """  # noqa: E501
        # Initialize the parent class
        super(H5SeqDataset, self).__init__(
            path=path,
            response_type=response_type,
            is_train=is_train,
            is_rgb=is_rgb,
            y_scaler=y_scaler,
        )
        self.dataset_len: int = self.dataset_len - seq_length
        self.seq_length: int = seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get the sequence of images
        x = self.X[idx : (idx + self.seq_length)]
        x = torch.from_numpy(x)

        # Apply image transformations
        x = self.transform_x(x)
        # Get one output value
        # Transform the output value to tensor
        y = torch.tensor(self.Y[:, idx + self.seq_length], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len


# Inspect the details of the H5SeqDataset class


if __name__ == "__main__":

    with open("config.yaml", "r") as stream:
        config = safe_load(stream)

    path = Path(config["DATA"]["path"])
    response_type = "firing_rate_10ms"
    is_train = True
    is_rgb = False
    y_scaler = None
    seq_length = 10

    dataset = H5SeqDataset(
        path=path,
        response_type=response_type,
        is_train=is_train,
        is_rgb=is_rgb,
        y_scaler=y_scaler,
        seq_length=seq_length,
    )

    X, y = dataset[0]
    print(f"Input data point shape: {X.shape}")
    print(f"Target data point shape: {y.shape}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence length: {dataset.seq_length}")
    print(f"Input shape: {dataset.input_shape}")
    print(f"Output shape: {dataset.output_shape}")
    print(f"Input type: {type(X)}")
    print(f"Output type: {type(y)}")
    print(f"Input dtype: {X.dtype}")
    print(f"Output dtype: {y.dtype}")
    print(f"Input min: {X.min()}")
    print(f"Input max: {X.max()}")
    print(f"Output min: {y.min()}")
    print(f"Output max: {y.max()}")
