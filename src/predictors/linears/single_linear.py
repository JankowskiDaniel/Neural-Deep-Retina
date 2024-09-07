from torch import nn
from interfaces import Predictor


class SingleLinear(Predictor):
    def __init__(self, input_size: int, num_classes: int):
        super(SingleLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        return x
