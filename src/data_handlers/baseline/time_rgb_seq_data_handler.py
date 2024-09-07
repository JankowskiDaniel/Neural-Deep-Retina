import torch
from typing import Any, Tuple
from pathlib import Path

from data_handlers import BaselineRGBDataset


class BaselineSeqRGBDataset(BaselineRGBDataset):
    def __init__(
        self,
        path: Path,
        response_type: str,
        results_dir: Path,
        is_train: bool = True,
        y_scaler: Any = None,
        seq_length: int = 10,
        use_saved_scaler: bool = False,
    ):
        super(BaselineSeqRGBDataset, self).__init__(
            path=path,
            response_type=response_type,
            is_train=is_train,
            y_scaler=y_scaler,
            results_dir=results_dir,
            use_saved_scaler=use_saved_scaler,
        )
        self.dataset_len: int = self.dataset_len - seq_length - 1
        self.seq_length: int = seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images_sequence = []

        for i in range(self.seq_length):
            images = []
            for j in range(3):
                x = self.X[idx + i + j]
                x = torch.from_numpy(x)
                images.append(x)

            stacked_image = torch.stack(images, dim=0)  # (3, H, W)
            images_sequence.append(stacked_image)

        # (sequence_length, 3, H, W)
        out = torch.stack(images_sequence, dim=0)
        out = self.transform_x(out)
        # Get the target for the last image in the sequence
        y = torch.tensor(self.Y[:, idx + self.seq_length + 1], dtype=torch.float32)

        return out, y

    def __len__(self):
        return self.dataset_len
