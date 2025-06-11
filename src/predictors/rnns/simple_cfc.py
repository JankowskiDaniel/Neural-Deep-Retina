from pathlib import Path
from torch import nn
from ncps.torch import CfC
import torch
from interfaces import Predictor


class SimpleCFC(Predictor):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
        hidden_size: int = 16,
        inner_activation: str = "lecun_tanh",
        output_activation: str | None = None,
        mixed_memory: bool = False,
        mode = "default",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
    ) -> None:
        super(SimpleCFC, self).__init__()
        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            proj_size=num_classes,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,
            activation=inner_activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            return_sequences=False,
        )
        self.activation = output_activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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