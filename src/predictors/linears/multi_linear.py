from pathlib import Path
from torch import nn
import torch
from interfaces import Predictor


class MultiLinear(Predictor):
    def __init__(
            self,
            input_size: int,
            num_classes: int,
            weights_path: Path | None = None,
    ) -> None:
        super(MultiLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, 64)
        self.lin2 = nn.Linear(64, num_classes)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x
