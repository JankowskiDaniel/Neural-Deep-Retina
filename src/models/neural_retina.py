import torch.nn as nn
from interfaces import Encoder, Predictor
from utils.torch_model_stats import count_parameters


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
        self.calculate_num_params()

    def calculate_num_params(self) -> None:
        """
        Calculate the number of trainable parameters in the model per component.
        """
        self.encoder_n_trainable_params = count_parameters(self.encoder)
        self.predictor_n_trainable_params = count_parameters(self.predictor)
        self.total_n_trainable_params = count_parameters(self)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x
