from torch import nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from interfaces import Predictor


class SimpleCFC(Predictor):
    def __init__(
        self, input_size: int, num_classes: int, hidden_size: int = 16
    ):
        super(SimpleCFC, self).__init__()
        wiring = AutoNCP(hidden_size, 12)
        self.cfc = CfC(
            input_size,
            wiring,
            batch_first=True,
            mixed_memory=True,
            return_sequences=False,
        )
        self.l1 = nn.Linear(12, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.cfc(x)
        x = self.l1(x)
        x = self.activation(x)
        return x
