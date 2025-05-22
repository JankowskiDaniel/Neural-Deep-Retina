from pathlib import Path
import torch.nn as nn
from interfaces import Predictor


class DecodingBlock(Predictor):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=nn.ELU(),
        kernel_size: int = 3,
    ) -> None:
        super(DecodingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                stride=1,
            ),
            activation,
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ShotSeqDecoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            out_channels: int = 40,
            weights_path: Path | None = None,
            num_classes: int = 1,
            activation: str = "relu",
            hidden_size: int = 32,
    ) -> None:
        super(ShotSeqDecoder, self).__init__()

        self.linear = nn.Linear(256, 16 * 3 * 3)
        self.unflatten = nn.Unflatten(1, (16, 3, 3))

        self.deconv = nn.Sequential(
            DecodingBlock(16, 64),  # 3x3 → 6x6
            DecodingBlock(64, 32),  # 6x6 → 12x12
            DecodingBlock(32, 16),  # 12x12 → 24x24
            DecodingBlock(16, out_channels),  # 24x24 → 48x48
        )

        # Final conv to adjust output shape to (40, 50, 50)
        self.final_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.upsample = nn.Upsample(
            size=(50, 50),
            mode='bilinear',
            align_corners=False
        )

    def forward(self, x):
        x = self.linear(x)                # (B, 32) → (B, 144)
        x = self.unflatten(x)            # (B, 144) → (B, 16, 3, 3)
        x = self.deconv(x)               # (B, 16, 3, 3) → (B, 40, 48, 48)
        x = self.upsample(x)             # (B, 40, 48, 48) → (B, 40, 50, 50)
        x = self.final_conv(x)           # Final refinement (optional)
        return x
