import warnings
import torch
from typing import Any, Literal, Tuple
from pathlib import Path
from interfaces.base_handler import BaseHandler


class BaselineRGBDataset(BaseHandler):
    def __init__(
        self,
        path: Path,
        response_type: Literal["firing_rate_10ms", "binned"],
        results_dir: Path,
        is_train: bool = True,
        y_scaler: Any = None,
        use_saved_scaler: bool = False,
        prediction_step: int = 0,
        subset_size: int = -1,
        **kwargs: Any,
    ) -> None:
        super(BaselineRGBDataset, self).__init__(
            path,
            response_type,
            results_dir,
            is_train,
            y_scaler,
            use_saved_scaler,
            prediction_step,
            subset_size,
        )

        self.subseq_length: int = 40
        self.dataset_len: int = self.dataset_len - self.subseq_length

        # List of allowed arguments in the constructor
        allowed_args = {
            "path",
            "response_type",
            "results_dir",
            "is_train",
            "y_scaler",
            "use_saved_scaler",
        }

        # Check for unused kwargs
        unused_kwargs = {k: v for k, v in kwargs.items() if k not in allowed_args}

        if unused_kwargs:
            # Print warning for unused kwargs
            warnings.warn(
                f"Unused arguments passed to the data handler: {unused_kwargs}. These will be ignored."  # noqa: E501
            )

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
        y = torch.tensor(
            self.Y[:, idx + self.subseq_length - 1 + self.prediction_step],
            dtype=torch.float32,
        )

        return x, y

    def __len__(self):
        return self.dataset_len
