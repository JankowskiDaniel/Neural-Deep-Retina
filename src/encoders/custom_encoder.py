from interfaces.encoder import Encoder

import torch.nn as nn
from torch import load, stack
from pathlib import Path


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
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            activation,
            # nn.BatchNorm2d(num_features=out_channels),
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
        self.flatten = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(4 * out_channels * 12 * 12, latent_dim),
            activation,
        )
        self.features = nn.Sequential(self.conv, self.flatten, self.bottleneck)

        self._output_shape = latent_dim

    def forward(self, x):
        if self.seq_len:
            latent_seq = []
            # batch
            for t in range(self.seq_len):
                x_t = self.features(x[:, t])
                x_t = x_t.view(x_t.size(0), -1)
                latent_seq.append(x_t)

            x = stack(latent_seq, dim=1)

        else:  # single image, not processed as a sequence
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor

        return x

    def load_weights_from_file(self, weights_path: Path) -> None:
        self.load_state_dict(load(weights_path))

    def get_output_shape(self):
        return self._output_shape
