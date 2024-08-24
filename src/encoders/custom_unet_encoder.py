import torch.nn as nn

from interfaces.encoder import Encoder
from .custom_encoder import EncodingBlock


class CustomUNETEncoder(Encoder):
    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomUNETEncoder, self).__init__()

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
