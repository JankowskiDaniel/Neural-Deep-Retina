import torch.nn as nn


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsampling=nn.Upsample(scale_factor=2, mode="nearest"),
        activation=nn.ReLU(),
    ) -> None:
        super(DecodingBlock, self).__init__()

        self.block = nn.Sequential(
            upsampling,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
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
        out_channels: int = 16,
        activation=nn.Sigmoid(),
    ) -> None:
        super(CustomDecoder, self).__init__()

        self.out_channels = out_channels

        image_channels = image_shape[0]

        self.transpose_conv = nn.Sequential(
            DecodingBlock(8 * out_channels, 4 * out_channels),
            DecodingBlock(4 * out_channels, 2 * out_channels),
            DecodingBlock(2 * out_channels, image_channels),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=image_channels,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            activation,
        )

    def forward(self, x):

        x = self.transpose_conv(x)
        return x
