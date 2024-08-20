from torch import nn
from interfaces import Predictor


class SingleLSTM(Predictor):
    def __init__(self, input_size: int, num_classes: int):
        super(SingleLSTM, self).__init__()
        self.flattened_size = input_size
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.lin1 = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.lin1(out[:, -1, :])
        return x
