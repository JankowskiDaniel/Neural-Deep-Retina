import torch
from torchvision import models
from pathlib import Path
from interfaces.encoder import Encoder


class VGG16Encoder(Encoder):
    def __init__(self,
                 input_shape: tuple,
                 weights_path: Path,
                 freeze: bool,
                 seq_len: int | None = None) -> None:
        super(VGG16Encoder, self).__init__()
        weights = models.vgg.VGG16_Weights
        weights.url = weights_path
        vgg16 = models.vgg16(weights=weights)
        self.features = vgg16.features
        self.seq_len = seq_len
        # Freeze the encoder
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Dummy input to determine the output shape
        self._dummy_input = torch.zeros(input_shape)
        self._output_shape = self._compute_output_shape()

    def forward(self, x):
        if self.seq_len:
            # VGG expects RGB input, so we may expect 5 values in the input shape
            # batch_size, seq_len, c, h, w = x.size()
            latent_seq = []

            for t in range(self.seq_len):
                x_t = self.features(x[:, t])
                x_t = x_t.view(x_t.size(0), -1) # (batch_size, 512)
                latent_seq.append(x_t)
            
            x = torch.stack(latent_seq, dim=1) # (batch_size, seq_len, 512)

        else: # single image, not processed as a sequence
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor

        return x

    def get_output_shape(self):
        return self._output_shape

    def _compute_output_shape(self):
        with torch.no_grad():
            output = self.features(self._dummy_input)
        return output.shape
