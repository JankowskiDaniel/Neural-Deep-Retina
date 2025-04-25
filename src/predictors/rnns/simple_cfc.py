from pathlib import Path
from torch import nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import torch
from interfaces import Predictor


class SimpleCFC(Predictor):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
        hidden_size: int = 16,
    ) -> None:
        super(SimpleCFC, self).__init__()
        wiring = AutoNCP(hidden_size, num_classes)
        self.cfc = CfC(
            input_size,
            wiring,
            batch_first=True,
            mixed_memory=True,
            return_sequences=False,
        )
        self.activation = nn.ReLU()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x, _ = self.cfc(x)
        x = self.activation(x)
        return x
