from interfaces.encoder import Encoder

import torch.nn as nn


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling=nn.MaxPool2d(kernel_size=2, stride=2),
        activation=nn.ReLU(),
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
            nn.Conv2d(
                in_channels=out_channels,
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
    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomEncoder, self).__init__()

        in_channels = image_shape[0]
        self.conv1 = EncodingBlock(in_channels, 2 * out_channels)
        self.conv2 = EncodingBlock(2 * out_channels, 4 * out_channels)
        self.conv3 = EncodingBlock(4 * out_channels, 8 * out_channels)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=8 * out_channels,
                out_channels=8 * out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            activation,
            nn.BatchNorm2d(num_features=8 * out_channels),
        )

        self._output_shape = (8 * out_channels, 6, 6)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = self.conv2(x)
        outputs.append(x)
        x = self.conv3(x)
        outputs.append(x)
        x = self.bottleneck(x)
        return x, outputs

    def get_output_shape(self):
        return self._output_shape