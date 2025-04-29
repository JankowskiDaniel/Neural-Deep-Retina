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
        activation: str | None = None,
    ):
        super(SingleLSTM, self).__init__()
        self.flattened_size = input_size
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            # proj_size=32,
        )
        self.dropout = nn.Dropout(0.2)  # manual dropout after LSTM
        self.norm1 = nn.LayerNorm(hidden_size)  # normalize after LSTM

        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.act0 = nn.ELU()

        self.l2 = nn.Linear(hidden_size, num_classes)
        self.act1 = nn.Identity()
        self.activation = activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Orthogonal initialization for LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                init.orthogonal_(param.data)
            elif "weight_hh" in name:  # Hidden-hidden weights
                init.orthogonal_(param.data)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.norm1(out)

        residual = out
        out = self.l1(out)
        out = self.act0(out)
        out = out + residual  # Residual connection

        out = self.l2(out)
        # out = self.act1(out)
        if self.activation is not None:
            if self.activation == "relu":
                x = self.relu(x)
            elif self.activation == "sigmoid":
                x = self.sigmoid(x)
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")
        return out
