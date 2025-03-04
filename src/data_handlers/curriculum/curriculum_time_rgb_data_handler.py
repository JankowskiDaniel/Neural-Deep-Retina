import warnings
import torch
from typing import Any, Literal, Tuple
from pathlib import Path
from interfaces.base_handler import BaseHandler


class CurriculumBaselineRGBDataset(BaseHandler):
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
        # Call parent init with scaler as None
        # This prevents the parent class from applying any scaling
        # As scaling will be done as part of curriculum learning
        super(CurriculumBaselineRGBDataset, self).__init__(
            path=path,
            response_type=response_type,
            is_train=is_train,
            subset_type=subset_type,
            y_scaler=None,
            results_dir=results_dir,
            use_saved_scaler=use_saved_scaler,
            prediction_step=prediction_step,
            subset_size=subset_size,
            pred_channels=pred_channels,
            is_classification=is_classification,
            class_epsilon=class_epsilon,
        )
        # Set y_scaler
        self.y_scaler = y_scaler

        # Transform to torch Tensor
        self.X: torch.Tensor = torch.from_numpy(self.X)
        self.curr_X = self.X

        # Apply the y-scaler to the target data
        if self.y_scaler is not None or self.use_saved_scaler:
            self.curr_Y = self.transform_y(self.Y.copy())
        else:
            self.curr_Y = self.Y.copy()
        # Transform to torch Tensor
        self.curr_Y = torch.from_numpy(self.curr_Y).to(torch.float32)

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

    def update_data(self, **kwargs) -> None:
        """
        Update the dataset by processing the data.
        """
        self.update_X(**kwargs)
        self.update_y(**kwargs)

    def update_X(self, **kwargs) -> None:
        pass

    def update_y(self, **kwargs) -> None:
        # Copy the original target data
        curr_Y = self.Y.copy()
        if "sigma" in kwargs:
            # Apply gaussian smoothing to the target data
            sigma = kwargs["sigma"]
            # TODO workaround for circular import
            from utils import (
                apply_gaussian_smoothening,
                # apply_asymmetric_gaussian_smoothening,
            )

            curr_Y = apply_gaussian_smoothening(curr_Y, sigma)
            # self.curr_Y = apply_asymmetric_gaussian_smoothening(self.curr_Y, sigma)

            # TODO: Remove this plot after debugging
            # fig, ax = plt.subplots(figsize=(20, 6))
            # ax.plot(self.curr_Y[1], label='gaussian filtered')
            # ax.set_title(f"Gaussian filtered channel 1")
            # plt.legend()
            # plt.savefig(f"gaussian_filtered_{sigma}.jpg", format="jpg", dpi=300)
            # plt.close()
        # Apply the y-scaler to the target data
        if self.y_scaler is not None:
            curr_Y = self.transform_y(curr_Y)

        # Transform to torch Tensor
        self.curr_Y = torch.from_numpy(curr_Y).to(torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.curr_X[idx : idx + self.subseq_len]
        # Apply any transformations
        x = self.transform_x(x)
        # Get the target for the fourth image
        y = self.curr_Y[:, idx + self.subseq_len - 1 + self.prediction_step]

        return x, y

    def __len__(self):
        return self.dataset_len
