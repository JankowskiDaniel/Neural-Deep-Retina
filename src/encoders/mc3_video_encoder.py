import torch
from pathlib import Path
from torchvision.models.video import resnet
from interfaces.encoder import Encoder


class MC3VideoEncoder(Encoder):
    def __init__(
        self,
        input_shape: tuple,
        weights_path: Path,
        freeze: bool,
        seq_len: int,
    ) -> None:
        super(MC3VideoEncoder, self).__init__()
        # weights = resnet.MC3_18_Weights
        # type(weights).KINETICS400_V1.url = str(weights_path)
        # print(weights)
        # weights.url = str(weights_path)
        mc3_18 = resnet.mc3_18()
        self.features = mc3_18
        mc3_18.stem[0] = torch.nn.Conv3d(
            seq_len,
            mc3_18.stem[0].out_channels,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.seq_len = seq_len
        # Freeze the encoder
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Dummy input to determine the output shape
        self._dummy_input = torch.zeros(input_shape)
        self._output_shape = self._compute_output_shape()

    def forward(self, x):
        x = self.features(x)
        return x

    def get_output_shape(self):
        return self._output_shape

    def _compute_output_shape(self):
        with torch.no_grad():
            output = self(self._dummy_input)
        return output.shape[-1]
