from pathlib import Path
from torch import nn
import torch
import torch.nn.init as init
from interfaces import Predictor


class SingleLSTM(Predictor):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        weights_path: Path | None = None,
        hidden_size: int = 16,
    ):
        super(SingleLSTM, self).__init__()
        self.flattened_size = input_size
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            proj_size=num_classes,
        )
        self.act = nn.Sigmoid()

        # Orthogonal initialization for LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                init.orthogonal_(param.data)
            elif "weight_hh" in name:  # Hidden-hidden weights
                init.orthogonal_(param.data)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.act(out[:, -1, :])
        return x
