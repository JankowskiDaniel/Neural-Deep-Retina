from interfaces.encoder import Encoder

import torch
import torch.nn as nn
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
            activation,
            nn.BatchNorm2d(num_features=out_channels),
            pooling,
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ShotSeqEncoder(Encoder):
    def __init__(
        self,
        input_shape: tuple,
        out_channels: int = 16,
        weights_path: Path | None = None,
        freeze: bool = False,
        seq_len: int = 1,
    ) -> None:
        super(ShotSeqEncoder, self).__init__()

        # The input shape is (batch_size, channels, height, width)
        # Channels is equal to the number of frames in the sequence
        self.seq_len = seq_len
        if self.seq_len >= 1:
            in_channels = input_shape[2]
        else:
            in_channels = input_shape[1]

        self.conv = nn.Sequential(
            EncodingBlock(in_channels, out_channels),
            EncodingBlock(out_channels, 2 * out_channels),
            EncodingBlock(2 * out_channels, 4 * out_channels),
            EncodingBlock(4 * out_channels, out_channels),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(out_channels * 3 * 3, 32)
        self.activation = nn.Tanh()
        self.bn1d = nn.BatchNorm1d(num_features=32)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self._dummy_input = torch.zeros(input_shape)
        self._output_shape = self._compute_output_shape()

    def forward(self, x):
        if self.seq_len >= 1:
            latent_seq = []
            # batch
            for t in range(self.seq_len):
                x_t = x[:, t]
                x_t = self.conv(x_t)
                x_t = self.flatten(x_t)
                x_t = self.linear(x_t)
                x_t = self.activation(x_t)
                x_t = x_t.view(x_t.size(0), -1)  # (batch_size, 32)
                latent_seq.append(x_t)

            # Concatenate the latent sequences along the sequence dimension
            x = torch.stack(latent_seq, dim=1)  # (batch_size, seq_len, 32)
        else:  # single image, not processed as a sequence
            x = self.conv(x)
            x = self.flatten(x)
            x = self.linear(x)
            x = self.activation(x)
            x = self.bn1d(x)
            # Add a dummy dimension for the sequence length
            x = x.unsqueeze(1)
        return x

    def get_output_shape(self):
        return self._output_shape

    def _compute_output_shape(self):
        with torch.no_grad():
            output = self(self._dummy_input)
        return output.shape[-1]
