from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.blocks import ConvBlock, FreqA


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)
        self.freqa = FreqA(out_channels, use_attention=use_attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.freqa(x)
        return x


class FAENet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        bottleneck_channels: int = 768,
        use_attention_in_shallow: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = nn.Sequential(ConvBlock(c1, c1), FreqA(c1, use_attention=use_attention_in_shallow))

        self.enc2 = nn.Sequential(ConvBlock(c1, c2), FreqA(c2, use_attention=False))

        self.enc3 = nn.Sequential(ConvBlock(c2, c3), FreqA(c3, use_attention=False))

        self.enc4 = nn.Sequential(ConvBlock(c3, c4), FreqA(c4, use_attention=True))

        self.bottleneck = nn.Sequential(
            ConvBlock(c4, bottleneck_channels),
            FreqA(bottleneck_channels, use_attention=True),
        )

        self.dec4 = DecoderStage(bottleneck_channels, c4, c4, use_attention=True)
        self.dec3 = DecoderStage(c4, c3, c3, use_attention=False)
        self.dec2 = DecoderStage(c3, c2, c2, use_attention=False)
        self.dec1 = DecoderStage(c2, c1, c1, use_attention=use_attention_in_shallow)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        e1 = self.enc1(x0)

        x1 = self.pool(e1)
        e2 = self.enc2(x1)

        x2 = self.pool(e2)
        e3 = self.enc3(x2)

        x3 = self.pool(e3)
        e4 = self.enc4(x3)

        x4 = self.pool(e4)
        b = self.bottleneck(x4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits = self.head(d1)
        return logits
