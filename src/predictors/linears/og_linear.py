from pathlib import Path
from torch import nn
import torch
from interfaces import Predictor


class OgLinear(Predictor):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
        hidden_size: int = 512,
        activation: str = "relu",
    ) -> None:
        super(OgLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, num_classes)
        self.bnd1 = nn.BatchNorm1d(num_classes, momentum=0.01, eps=1e-3)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.lin1(x)
        # x = self.bnd1(x)
        return x
