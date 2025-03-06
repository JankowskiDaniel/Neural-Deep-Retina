import warnings
import torch
from typing import Any, Tuple, Literal
from pathlib import Path

from interfaces.base_handler import BaseHandler


class BaselineSeqRGBDataset(BaseHandler):
    def __init__(
        self,
        seq_len: int,
        path: Path,
        subseq_len: int,
        response_type: str,
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
    ):
        super(BaselineSeqRGBDataset, self).__init__(
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
        self.dataset_len: int = (
            self.dataset_len - seq_len - 1 - self.subseq_len
        )
        self.seq_length: int = seq_len

        # List of allowed arguments in the constructor
        allowed_args = {
            "path",
            "response_type",
            "results_dir",
            "is_train",
            "y_scaler",
            "use_saved_scaler",
            "seq_len",
            "subseq_len",
        }

        # Check for unused kwargs
        unused_kwargs = {
            k: v for k, v in kwargs.items() if k not in allowed_args
        }

        if unused_kwargs:
            warnings.warn(
                f"Unused arguments passed to the data handler: {unused_kwargs}. These will be ignored."  # noqa: E501
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images_sequence = []

        for i in range(self.seq_length):
            images = []
            for j in range(self.subseq_len):
                x = self.X[idx + i + j]
                x = torch.from_numpy(x)
                images.append(x)

            stacked_image = torch.stack(images, dim=0)  # (3, H, W)
            images_sequence.append(stacked_image)

        # (sequence_length, subseq_len, H, W)
        out = torch.stack(images_sequence, dim=0)
        out = self.transform_x(out)
        # Get the target for the last image in the sequence
        y = torch.tensor(
            self.Y[:, idx + self.seq_length + 1 + self.prediction_step],
            dtype=torch.float32,
        )
        return out, y

    def __len__(self):
        return self.dataset_len
