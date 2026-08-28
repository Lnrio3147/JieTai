"""One-step direct clean-mask predictor inspired by JiT's x-prediction.

This is deliberately not a diffusion model. It borrows the useful inductive
biases (direct clean-data prediction, large raw patches, low-rank bottleneck)
while retaining a single RKNN-friendly convolutional inference pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LargePatchLowRankBottleneck(nn.Module):
    """Compress large spatial patches before projecting back to the feature grid."""

    def __init__(self, channels: int, bottleneck_channels: int, patch_size: int) -> None:
        super().__init__()
        self.patch_reduce = nn.Sequential(
            nn.Conv2d(
                channels,
                bottleneck_channels,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU6(inplace=True),
        )
        self.patch_mixer = nn.Sequential(
            DepthwiseSeparableConv(
                bottleneck_channels, bottleneck_channels, kernel_size=5
            ),
            DepthwiseSeparableConv(
                bottleneck_channels, bottleneck_channels, kernel_size=3
            ),
        )
        self.expand = ConvBNAct(bottleneck_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = self.expand(self.patch_mixer(self.patch_reduce(x)))
        return F.interpolate(
            low_rank, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class UpAddBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_projection = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.skip_projection = ConvBNAct(skip_channels, out_channels, kernel_size=1)
        self.refine = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.input_projection(x) + self.skip_projection(skip))


class CleanMaskProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        channels: Sequence[int] = (16, 24, 32, 48),
        bottleneck_channels: int = 16,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (int(value) for value in channels)
        self.stem = ConvBNAct(in_channels, c1)
        self.down1 = DepthwiseSeparableConv(c1, c2, stride=2)
        self.down2 = DepthwiseSeparableConv(c2, c3, stride=2)
        self.down3 = DepthwiseSeparableConv(c3, c4, stride=2)
        self.bottleneck = LargePatchLowRankBottleneck(
            c4, bottleneck_channels, patch_size
        )
        self.up3 = UpAddBlock(c4, c3, c3)
        self.up2 = UpAddBlock(c3, c2, c2)
        self.up1 = UpAddBlock(c2, c1, c1)
        self.correction_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)
        # Epoch zero is exactly the Exp8 Base mask. Training is allowed to learn
        # only evidence-backed residual corrections rather than rebuilding it.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_probability = features[:, 6:7].clamp(1e-4, 1.0 - 1e-4)
        coarse_logits = torch.logit(coarse_probability)
        s1 = self.stem(features)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x = self.bottleneck(self.down3(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        clean_logits = coarse_logits + self.correction_head(x)
        return clean_logits, self.boundary_head(x)


def create_projector(**kwargs) -> CleanMaskProjector:
    return CleanMaskProjector(**kwargs)


if __name__ == "__main__":
    model = create_projector().eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 9, 512, 288))
    print([tuple(value.shape) for value in output])
    print(sum(parameter.numel() for parameter in model.parameters()))
