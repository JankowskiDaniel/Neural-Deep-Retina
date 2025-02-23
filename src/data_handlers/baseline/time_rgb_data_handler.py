import warnings
import torch
from typing import Any, Literal, Tuple
from pathlib import Path
from interfaces.base_handler import BaseHandler


class BaselineRGBDataset(BaseHandler):
    def __init__(
        self,
        path: Path,
        subseq_len: int,
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
        super(BaselineRGBDataset, self).__init__(
            path=path,
            response_type=response_type,
            is_train=is_train,
            subset_type=subset_type,
            y_scaler=y_scaler,
            results_dir=results_dir,
            use_saved_scaler=use_saved_scaler,
            prediction_step=prediction_step,
            subset_size=subset_size,
            pred_channels=pred_channels,
            is_classification=is_classification,
            class_epsilon=class_epsilon,
        )

        self.subseq_len: int = subseq_len
        self.dataset_len: int = self.dataset_len - self.subseq_len

        # List of allowed arguments in the constructor
        allowed_args = {
            "path",
            "subseq_len",
            "response_type",
            "results_dir",
            "is_train",
            "subset_type",
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
        for i in range(self.subseq_len):
            x = self.X[idx + i]
            x = torch.from_numpy(x)
            images.append(x)
        # Stack images along the channel dimension
        x = torch.stack(images, dim=0)  # Shape will be (3, H, W)
        # Apply any transformations to the stacked images
        x = self.transform_x(x)
        # Get the target for the fourth image
        y = torch.tensor(
            self.Y[:, idx + self.subseq_len - 1 + self.prediction_step],
            dtype=torch.float32,
        )

        return x, y

    def __len__(self):
        return self.dataset_len
