from pathlib import Path
import torch.nn as nn
import torch


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


class OgEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        weights_path: Path | None,
        freeze: bool,
        seq_len: int,
    ) -> None:
        super(OgEncoder, self).__init__()
        self.seq_len = seq_len

        # Dummy input to determine the output shape
        if self.seq_len >= 1:
            self.in_channels = input_shape[2]
            self.width = input_shape[3]
            self.height = input_shape[4]
        else:
            self.in_channels = input_shape[1]
            self.width = input_shape[2]
            self.height = input_shape[3]
        self.out_channels = 8

        # Declare a model with 3 2D-Conv layers
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=15,
            stride=1,
            padding=0,
        )
        self.flat1 = Flatten()
        self.bn1d1 = nn.BatchNorm1d(
            self.out_channels * 36 * 36,
            momentum=0.01,
            eps=1e-3,
        )
        self.reshape1 = Reshape(
            (-1, self.out_channels, 36, 36)
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels * 2,
            kernel_size=11,
            stride=1,
            padding=0,
        )
        self.flat2 = Flatten()
        self.bn1d2 = nn.BatchNorm1d(
            self.out_channels * 2 * 26 * 26,
            momentum=0.01,
            eps=1e-3,
        )
        self.reshape2 = Reshape(
            (-1, self.out_channels * 2, 26, 26)
        )
        self.flat3 = Flatten()

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
                x_t = self.conv1(x_t)
                x_t = self.flat1(x_t)
                x_t = self.bn1d1(x_t)
                x_t = self.reshape1(x_t)

                x_t = self.conv2(x_t)
                x_t = self.flat2(x_t)
                x_t = self.bn1d2(x_t)
                x_t = self.reshape2(x_t)
                x_t = self.flat3(x_t)

                x_t = x_t.view(x_t.size(0), -1)  # (batch_size, 512)

                latent_seq.append(x_t)

            x = torch.stack(latent_seq, dim=1)  # (batch_size, seq_len, 512)

        else:  # single image, not processed as a sequence
            x = self.conv1(x)
            x = self.flat1(x)
            x = self.bn1d1(x)
            x = self.reshape1(x)
            x = nn.ReLU()(x)

            x = self.conv2(x)
            x = self.flat2(x)
            x = self.bn1d2(x)
            x = self.reshape2(x)
            x = nn.ReLU()(x)
            x = self.flat3(x)

        # print(x.shape)
        return x

    def get_output_shape(self):
        return self._output_shape

    def _compute_output_shape(self):
        with torch.no_grad():
            output = self(self._dummy_input)
        return output.shape[-1]
