from torch import nn
from interfaces import Predictor


class MultiLinear(Predictor):
    def __init__(self, input_size: int, num_classes: int):
        super(MultiLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, 64)
        self.lin2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x
