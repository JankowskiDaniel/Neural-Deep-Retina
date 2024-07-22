from torch import nn
from interfaces.encoder import Encoder


class DummyCNN(nn.Module):
    def __init__(self, encoder: Encoder, num_classes: int):
        super(DummyCNN, self).__init__()
        self.encoder = encoder
        output_shape = encoder.get_output_shape()
        flattened_size = output_shape[1] * output_shape[2] * output_shape[3]
        self.lin1 = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.lin1(x)
        return x
