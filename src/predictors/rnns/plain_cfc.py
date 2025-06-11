from pathlib import Path
from torch import nn
from ncps.torch import CfC
import torch
from interfaces import Predictor


class PlainCFC(Predictor):
    """
    A simple implementation of the CfC predictor
    without the AutoNCP wiring.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
        hidden_size: int = 16,
        activation: str | None = None,
    ) -> None:
        super(PlainCFC, self).__init__()
        self.cfc = CfC(
            input_size,
            hidden_size,
            proj_size=num_classes,
            batch_first=True,
            mixed_memory=False,
            return_sequences=False,
            activation="gelu",
            backbone_units=256,
            backbone_layers=3,
        )
        self.activation = activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.constant_(self.cfc.fc.bias, -4.0)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x, _ = self.cfc(x)
        if self.activation is not None:
            if self.activation == "relu":
                x = self.relu(x)
            elif self.activation == "sigmoid":
                x = self.sigmoid(x)
            else:
                raise ValueError(
                    f"Unknown activation function: {self.activation}"
                )
        return x
