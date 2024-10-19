from torch import nn
import torch.nn.init as init
from interfaces import Predictor


class SingleLSTM(Predictor):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 32):
        super(SingleLSTM, self).__init__()
        self.flattened_size = input_size
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.lin1 = nn.Linear(hidden_size, num_classes)
        self.bnd1 = nn.BatchNorm1d(num_classes, momentum=0.01, eps=1e-3)
        self.softplus = nn.Softplus()

        # Orthogonal initialization for LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                init.orthogonal_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                init.orthogonal_(param.data)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.lin1(out[:, -1, :])
        x = self.bnd1(x)
        x = self.softplus(x)
        return x
