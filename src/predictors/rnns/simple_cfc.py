from torch import nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from interfaces import Predictor


class SimpleCFC(Predictor):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 16):
        super(SimpleCFC, self).__init__()
        wiring = AutoNCP(hidden_size, num_classes)
        self.cfc = CfC(
            input_size,
            wiring,
            batch_first=True,
            mixed_memory=True,
            return_sequences=False,
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.cfc(x)
        # x = self.softplus(x)
        return x
