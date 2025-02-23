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


class BaseCustomEncoder(Encoder):
    """A basic encoder for the autoencoder. No copy&crop connections."""

    def __init__(
        self,
        out_channels: int = 4,
        latent_dim: int = 100,
        activation=nn.ReLU(),
    ) -> None:
        super(BaseCustomEncoder, self).__init__()

        in_channels = 1
        self.conv = nn.Sequential(
            EncodingBlock(in_channels, 2 * out_channels),
            EncodingBlock(2 * out_channels, 4 * out_channels),
            EncodingBlock(4 * out_channels, 8 * out_channels),
        )
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(4 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
        #     activation,
        # )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * out_channels * 6 * 6, latent_dim),
            activation,
            nn.BatchNorm1d(num_features=latent_dim),
        )

        self._output_shape = latent_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.bottleneck(x)
        return x

    def get_output_shape(self):
        return self._output_shape
