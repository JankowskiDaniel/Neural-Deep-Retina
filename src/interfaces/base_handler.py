import torch
from h5py import File
from torchvision.transforms import v2
import numpy as np
from numpy import ndarray, dtype
from typing import Any, Literal, Tuple
from pathlib import Path
import pickle
from abc import abstractmethod

transform_x: v2.Compose = v2.Compose([v2.ToDtype(torch.float32, scale=True)])


class BaseHandler(torch.utils.data.Dataset):
    """
    Initialize the base handler.

    Args:
        path (Path): Path to the dataset file.
        response_type (Literal["firing_rate_10ms", "binned"]): Type of response data.
        results_dir (Path): Directory to save results.
        is_train (bool, optional): Flag to indicate if training or testing data should be read. Defaults to True.
        y_scaler (Any, optional): Scaler for the response variable. Defaults to None.
        use_saved_scaler (bool, optional): Flag to use a saved scaler for training data. Defaults to False.
        prediction_step (int, optional): Step size for prediction. Defaults to 0.
        subset_size (int, optional): Size of the subset to use. Defaults to -1 (full dataset).
        **kwargs (Any): Additional keyword arguments.

    Returns:
        None
    """  # noqa: E501

    def __init__(
        self,
        path: Path,
        response_type: Literal["firing_rate_10ms", "binned"],
        results_dir: Path,
        is_train: bool = True,
        subset_type: Literal["train", "val", "test"] = "train",
        y_scaler: Any = None,
        use_saved_scaler: bool = False,
        prediction_step: int = 0,
        subset_size: int = -1,
        pred_channels: list[int] = [],
        is_classification: bool = False,
        class_epsilon: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.file_path = path
        # The available types are firing_rate_10ms, binned
        self.response_type = response_type
        self.is_train = is_train
        self.subset_type = subset_type
        self.y_scaler = y_scaler
        self.results_dir = results_dir
        self.transform_x = transform_x
        # Allows to use the saved scaler for the train data
        self.use_saved_scaler = use_saved_scaler
        self.subset_size: int = subset_size

        # Classification parameters
        self.is_classification: bool = is_classification
        self.class_epsilon: float = class_epsilon

        self.pred_channels: list[int] = pred_channels
        # Read dataset from file
        X, y = self.read_h5_to_numpy()
        # Truncate the data if subset_size is provided
        X, y = self.truncate_data(X, y)
        # Select the channels
        y = self.select_channels(y)
        # Normalize the output data
        if self.y_scaler is not None or self.use_saved_scaler:
            y = self.transform_y(y)
        # Optional binarization of the output data
        if self.is_classification:
            y = self.binarize(y)
        self.prediction_step: int = prediction_step
        self.dataset_len = len(X) - self.prediction_step
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
            X = np.asarray(h5file[self.subset_type]["stimulus"])
            y = np.asarray(
                h5file[self.subset_type]["response"][self.response_type]
            )
        y = y.astype("float32")

        return X, y

    def get_target(self) -> ndarray[Any, dtype[Any]]:
        """
        Returns the target variable 'y'.

        Returns:
        - ndarray[Any, dtype[Any]]
            The target variable.
        """
        return self.Y

    def truncate_data(
        self, X: ndarray[Any, dtype[Any]], y: ndarray[Any, dtype[Any]]
    ) -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        # Subset the data if positive subset_size is provided
        if self.subset_size > 0:
            X = X[: self.subset_size]
            y = y[:, : self.subset_size]
        return X, y

    def select_channels(
        self, y: ndarray[Any, dtype[Any]]
    ) -> ndarray[Any, dtype[Any]]:
        return y[self.pred_channels]

    def transform_y(
        self, y: ndarray[Any, dtype[Any]]
    ) -> ndarray[Any, dtype[Any]]:
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
                print("Saving scaler", self.y_scaler.data_range_)
        else:
            try:
                # load the scaler
                with open(self.results_dir / "y_scaler.pkl", "rb") as f:
                    self.y_scaler = pickle.load(f)
                # Only transform the test data
                y_fit = self.y_scaler.transform(y_tran)
            except FileNotFoundError:
                print(
                    "The scaler file is not found. Target will not be scaled"
                )
                y_fit = y_tran
        y = y_fit.T  # return the data to the original shape
        return y

    def get_y_scaler(self) -> Any:
        """
        Returns the y_scaler object.

        Returns:
        - Any
            The y_scaler object.
        """
        return self.y_scaler

    def binarize(
        self, y: ndarray[Any, dtype[Any]]
    ) -> ndarray[Any, dtype[Any]]:
        return np.where(y >= self.class_epsilon, 1, 0)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self):
        return self.dataset_len
