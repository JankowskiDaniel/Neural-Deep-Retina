import torch.nn as nn
from torch import cat
from torchvision.transforms.functional import center_crop

from .custom_decoder import DecodingBlock


class CustomUNETDecoder(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 16,
        activation=nn.Sigmoid(),
    ) -> None:
        super(CustomUNETDecoder, self).__init__()

        self.out_channels = out_channels

        image_channels = image_shape[0]

        self.transpose_conv1 = DecodingBlock(8 * out_channels, 4 * out_channels)
        self.transpose_conv2 = DecodingBlock(4 * out_channels, 2 * out_channels)
        self.transpose_conv3 = DecodingBlock(2 * out_channels, image_channels)
        self.conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=image_channels,
            kernel_size=1,
            stride=1,
            padding=1,
        )
        self.activation = activation

    def forward(self, x, outputs: list):

        # outputs are the intermediate outputs from the encoder
        out1 = outputs.pop(-1)
        out1 = center_crop(out1, x.shape[2])
        x = cat([x, out1], dim=1)
        x = self.transpose_conv1(x)

        out2 = outputs.pop(-1)
        out2 = center_crop(out2, x.shape[2])
        x = cat([x, out2], dim=1)
        x = self.transpose_conv2(x)

        out3 = outputs.pop(-1)
        out3 = center_crop(out3, x.shape[2])
        x = cat([x, out3], dim=1)
        x = self.transpose_conv3(x)

        x = self.conv(x)
        x = self.activation(x)
        return x
