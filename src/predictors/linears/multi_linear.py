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
        hidden_size: int = 64,
    ) -> None:
        super(MultiLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, hidden_size)
        self.bnd1 = nn.BatchNorm1d(hidden_size, momentum=0.01, eps=1e-3)
        self.act0 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.act1 = nn.ReLU()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.lin1(x)
        x = self.act0(x)
        x = self.bnd1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.act1(x)
        return x
