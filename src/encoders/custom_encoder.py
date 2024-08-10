import torch.nn as nn

from interfaces.encoder import Encoder


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling=nn.MaxPool2d(2),
        activation=nn.ReLU(),
    ) -> None:
        super(EncodingBlock, self).__init__()

        modules = []
        modules.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        modules.append(activation)
        modules.append(nn.BatchNorm2d(num_features=in_channels))
        modules.append(pooling)

        self.block = nn.Sequential(*modules)

    def forward(self, input):
        x = self.block(input)
        return x


class CustomEncoder(Encoder):
    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 16,
        latent_dim: int = 100,
        activation=nn.ELU(),
    ) -> None:
        super(CustomEncoder, self).__init__()

        in_channels = image_shape[0]
        self.net = nn.Sequential(
            EncodingBlock(in_channels, 2 * out_channels),
            EncodingBlock(2 * out_channels, 4 * out_channels),
            EncodingBlock(4 * out_channels, 8 * out_channels),
            nn.Flatten(),
            nn.Linear(8 * out_channels * 7 * 7, latent_dim),
            activation,
        )

        self._output_shape = latent_dim

    def forward(self, x):
        return self.net(x)

    def get_output_shape(self):
        return self._output_shape
