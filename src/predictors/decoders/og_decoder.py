from pathlib import Path
import torch.nn as nn
from interfaces import Predictor


class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Reshape(nn.Module):
    """
    Reshapes the activations to be of shape (B, *shape) where B
    is the batch size.
    """

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)


class OgDecoder(Predictor):
    def __init__(
        self,
        input_size: int,
        out_channels: int = 40,
        weights_path: Path | None = None,
        num_classes: int = 1,
        activation: str = "relu",
        hidden_size: int = 32,
        latent_dim: int = 10816,
    ) -> None:
        """
        Mirrors the OgEncoder to reconstruct images of shape [40, 50, 50].
        latent_dim = final flattened size from the encoder = 16 x 26 x 26 = 10816
        """
        super(OgDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = out_channels

        # Same shape as after Conv2 in encoder
        self.reshape1 = Reshape((-1, 16, 26, 26))
        self.bn1 = nn.BatchNorm2d(16, momentum=0.01, eps=1e-3)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=11,
            stride=1,
            padding=0,
        )

        self.bn2 = nn.BatchNorm2d(8, momentum=0.01, eps=1e-3)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=out_channels,
            kernel_size=15,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        """
        x: (batch_size, latent_dim)  e.g., (B, 10816)
        Returns:
            Reconstructed images of shape (B, 40, 50, 50)
        """
        x = x.view(x.size(0), self.latent_dim)
        x = self.reshape1(x)  # (B, 16, 26, 26)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.deconv1(x)   # (B, 8, 36, 36)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        x = self.deconv2(x)   # (B, 40, 50, 50)

        return x
