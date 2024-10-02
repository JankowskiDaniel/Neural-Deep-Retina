from torch import nn
from interfaces import Predictor


class OgLinear(Predictor):
    def __init__(self, input_size: int, num_classes: int):
        super(OgLinear, self).__init__()
        self.flattened_size = input_size
        self.lin1 = nn.Linear(self.flattened_size, num_classes)
        self.bnd1 = nn.BatchNorm1d(num_classes, momentum=0.01, eps=1e-3)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.lin1(x)
        x = self.bnd1(x)
        x = self.softplus(x)
        return x