from interfaces.encoder import Encoder
from encoders.base_custom_encoder import BaseCustomEncoder

import torch.nn as nn
from torch import load, stack, zeros, no_grad
from pathlib import Path


class CustomEncoder(Encoder):
    """A basic encoder for the autoencoder. No copy&crop connections."""

    def __init__(
        self,
        input_shape: tuple,
        weights_path: Path,
        freeze: bool,
        seq_len: int,
        out_channels: int = 4,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomEncoder, self).__init__()

        base_custom_encoder = BaseCustomEncoder(
            out_channels=out_channels, activation=activation
        )
        base_custom_encoder.load_state_dict(load(weights_path))

        self.features = base_custom_encoder

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        # Dummy input to determine the output shape
        self._dummy_input = zeros(input_shape)
        self._output_shape = self._compute_output_shape()

    def forward(self, x):
        if self.seq_len >= 1:
            latent_seq = []
            # batch
            for t in range(self.seq_len):
                x_t = x[:, t]
                x_t = self.features(x_t)
                x_t = x_t.view(x_t.size(0), -1)
                latent_seq.append(x_t)

            x = stack(latent_seq, dim=1)

        else:  # single image, not processed as a sequence
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor

        # Scale the output
        x = x * 0.125

        return x

    def get_output_shape(self):

        return self._output_shape

    def _compute_output_shape(self):
        with no_grad():
            output = self(self._dummy_input)
        return output.shape[-1]
