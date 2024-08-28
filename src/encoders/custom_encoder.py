from interfaces.encoder import Encoder
from encoders.base_custom_encoder import BaseCustomEncoder

import torch.nn as nn
from torch import load, stack
from pathlib import Path


class CustomEncoder(Encoder):
    """A basic encoder for the autoencoder. No copy&crop connections."""

    def __init__(
        self,
        input_shape: tuple,
        weights_path: Path,
        freeze: bool,
        seq_len: int,
        latent_dim: int = 100,
        out_channels: int = 16,
        activation=nn.ReLU(),
    ) -> None:
        super(CustomEncoder, self).__init__()

        base_custom_encoder = BaseCustomEncoder(
            input_shape, latent_dim, out_channels, activation
        )
        base_custom_encoder.load_state_dict(load(weights_path))

        self.features = base_custom_encoder

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        self._output_shape = latent_dim

    def forward(self, x):
        if self.seq_len > 1:
            latent_seq = []
            # batch
            for t in range(self.seq_len):
                x_t = self.features(x[:, t].unsqueeze(1))
                x_t = x_t.view(x_t.size(0), -1)
                latent_seq.append(x_t)

            x = stack(latent_seq, dim=1)

        else:  # single image, not processed as a sequence
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor

        return x

    def get_output_shape(self):
        return self._output_shape
