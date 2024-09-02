import torch.nn as nn


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsampling=nn.Upsample(scale_factor=2, mode="bilinear"),
        activation=nn.ELU(),
    ) -> None:
        super(DecodingBlock, self).__init__()

        self.block = nn.Sequential(
            upsampling,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            activation,
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CustomDecoder(nn.Module):
    """A basic decoder for the autoencoder. No copy&crop connections."""

    def __init__(
        self,
        image_shape: tuple,
        latent_dim: int,
        out_channels: int = 16,
        activation=nn.Sigmoid(),
    ) -> None:
        super(CustomDecoder, self).__init__()

        # The number of intermediate channels in
        # convolutional layers, must match the encoder
        self.out_channels = out_channels

        image_channels = image_shape[0]

        self.transpose_conv = nn.Sequential(
            DecodingBlock(4 * out_channels, 2 * out_channels),
            DecodingBlock(2 * out_channels, image_channels),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=image_channels,
                kernel_size=3,
                padding=2,
                stride=1,
            ),
            nn.ReLU(),
        )

        self._output_shape = latent_dim

    def forward(self, x):
        x = self.transpose_conv(x)
        return x
