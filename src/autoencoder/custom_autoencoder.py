import torch.nn as nn
from encoders.custom_encoder import CustomEncoder
from .custom_decoder import CustomDecoder


class CustomAutoencoder(nn.Module):
    """
    A custom conv-dense autoencoder. It is composed of an encoder and a
    decoder. The purpose of this class is to easily combine the encoder and
    decoder into a single model.
    """

    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ):
        super(CustomAutoencoder, self).__init__()
        # Initialize the encoder and decoder
        self.encoder = CustomEncoder(image_shape, out_channels, activation)
        self.decoder = CustomDecoder(image_shape, out_channels, activation)

    def forward(self, x):
        x = self.encoder(x)
        self.latent = x
        x = self.decoder(x)
        return x
