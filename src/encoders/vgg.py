from torchvision import models
import torch
from interfaces.encoder import Encoder


class VGG16Encoder(Encoder):
    def __init__(self, input_shape: tuple, freeze: bool) -> None:
        super(VGG16Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features

        # Freeze the encoder
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Dummy input to determine the output shape
        self._dummy_input = torch.zeros(input_shape)
        self._output_shape = self._compute_output_shape()

    def forward(self, x):
        return self.features(x)

    def get_output_shape(self):
        return self._output_shape

    def _compute_output_shape(self):
        with torch.no_grad():
            output = self.features(self._dummy_input)
        return output.shape
