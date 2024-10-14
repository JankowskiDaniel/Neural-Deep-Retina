import torch
from typing import Any, Literal, Tuple
from pathlib import Path
from interfaces.base_handler import BaseHandler
import warnings


class H5Dataset(BaseHandler):
    def __init__(
        self,
        is_rgb: bool,
        path: Path,
        response_type: Literal["firing_rate_10ms", "binned"],
        results_dir: Path,
        is_train: bool = True,
        y_scaler: Any = None,
        use_saved_scaler: bool = False,
        prediction_step: int = 0,
        **kwargs: Any,
    ) -> None:
        super(H5Dataset, self).__init__(
            path,
            response_type,
            results_dir,
            is_train,
            y_scaler,
            use_saved_scaler,
            prediction_step,
        )
        self.is_rgb = is_rgb

        # List of allowed arguments in the constructor
        allowed_args = {
            "path",
            "response_type",
            "results_dir",
            "is_train",
            "y_scaler",
            "use_saved_scaler",
            "is_rgb",
        }

        # Check for unused kwargs
        unused_kwargs = {k: v for k, v in kwargs.items() if k not in allowed_args}

        if unused_kwargs:
            # Print warning for unused kwargs
            warnings.warn(
                f"Unused arguments passed to the data handler: {unused_kwargs}. These will be ignored."  # noqa: E501
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.X[idx]
        x = torch.from_numpy(x)
        if self.is_rgb:
            x = x.repeat(3, 1, 1)

        # Transform the image to tensor
        x = self.transform_x(x)
        # Transform the output value to tensor
        y = torch.tensor(self.Y[:, idx + self.prediction_step], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.dataset_len
