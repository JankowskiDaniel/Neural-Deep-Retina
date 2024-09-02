from interfaces.encoder import Encoder

import torch.nn as nn


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling=nn.MaxPool2d(kernel_size=2, stride=2),
        activation=nn.ELU(),
    ) -> None:
        super(EncodingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            activation,
            nn.BatchNorm2d(num_features=out_channels),
            pooling,
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CustomEncoder(Encoder):
    """A basic encoder for the autoencoder. No copy&crop connections."""

    def __init__(
        self,
        image_shape: tuple,
        latent_dim: int = 100,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomEncoder, self).__init__()

        in_channels = image_shape[0]
        self.conv = nn.Sequential(
            EncodingBlock(in_channels, 2 * out_channels),
            EncodingBlock(2 * out_channels, 4 * out_channels),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(4 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
            activation,
        )

        self._output_shape = latent_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.bottleneck(x)
        return x

    def get_output_shape(self):
        return self._output_shape
