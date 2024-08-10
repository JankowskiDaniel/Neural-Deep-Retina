import torch.nn as nn


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=nn.ReLU(),
    ) -> None:
        super(DecodingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            activation,
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CustomDecoder(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        latent_dim: int,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomDecoder, self).__init__()

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 8 * out_channels * 6 * 6),
            activation,
        )

        image_channels = image_shape[0]

        self.transpose_conv = nn.Sequential(
            DecodingBlock(8 * out_channels, 4 * out_channels),
            DecodingBlock(4 * out_channels, 2 * out_channels),
            DecodingBlock(2 * out_channels, image_channels),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=image_channels,
                kernel_size=3,
                padding=2,
                stride=1,
            ),
            activation,
        )

        self._output_shape = latent_dim

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 8 * self.out_channels, 6, 6)
        x = self.transpose_conv(x)
        return x
