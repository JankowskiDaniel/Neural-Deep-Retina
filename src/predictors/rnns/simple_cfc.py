from torch import nn
from ncps.torch import CfC
from interfaces import Predictor


class SimpleCFC(Predictor):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 8):
        super(SimpleCFC, self).__init__()
        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            batch_first=True,
            proj_size=num_classes,
            return_sequences=False,
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.cfc(x)
        # x = self.softplus(x)
        return x
