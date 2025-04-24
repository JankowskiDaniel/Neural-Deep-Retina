from pathlib import Path
from torch import nn
import torch
from interfaces import Predictor


class SingleLinear(Predictor):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
    ) -> None:
        super(SingleLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, num_classes)
        self.softplus = nn.Softplus()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.lin1(x)
        x = self.softplus(x)
        return x
