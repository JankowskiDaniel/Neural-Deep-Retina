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
        hidden_size: int = 64,
        activation: str | None = None,
    ) -> None:
        super(SingleLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, num_classes)
        # Initialize the bias
        torch.nn.init.constant_(self.lin1.bias, -4.0)
        self.act1 = nn.Sigmoid() if activation == "sigmoid" else nn.ReLU()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.act1(x)
        return x
