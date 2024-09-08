from torch import nn
from interfaces import Predictor


class SingleLSTM(Predictor):
    def __init__(
        self, input_size: int, num_classes: int, hidden_size: int = 8
    ):
        super(SingleLSTM, self).__init__()
        self.flattened_size = input_size
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.lin1 = nn.Linear(hidden_size, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.lin1(out[:, -1, :])
        x = self.softplus(x)
        return x
