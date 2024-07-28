import torch.nn as nn
from interfaces import Encoder, Predictor


class DeepRetinaModel(nn.Module):
    """
    The final architecture of the model. It is composed of an encoder and a
    predictor. The purpose of this class is to easily combine the encoder and
    predictor into a single model.
    """
    def __init__(self, encoder: Encoder, predictor: Predictor):
        super(DeepRetinaModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.predictor(x)
        return x
