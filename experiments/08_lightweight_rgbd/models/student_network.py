"""MobileNetV4 + shallow geometry gating + EMCAD student network.

The deployment graph intentionally uses standard convolution, depthwise
convolution, BatchNorm, ReLU6, reductions, sigmoid, multiplication and resize
operations. These operators are substantially easier to convert with RKNN than
deformable attention or transformer token operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import timm
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
                stride=stride,
                padding=padding,
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
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=True,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ShallowGeometryEncoder(nn.Module):
    """Exactly three learned convolutional layers for disparity geometry.

    The input channels are robust-normalized disparity, Sobel magnitude and a
    validity map. Learned features stop at 1/8; parameter-free average pooling
    supplies the 1/16 and 1/32 geometry scales.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ConvBNAct(3, 16, 3, stride=2)
        self.layer2 = DepthwiseSeparableConv(16, 24, 3, stride=2)
        self.layer3 = DepthwiseSeparableConv(24, 32, 3, stride=2)

    def forward(self, geometry: torch.Tensor) -> list[torch.Tensor]:
        x = self.layer1(geometry)
        scale4 = self.layer2(x)
        scale8 = self.layer3(scale4)
        scale16 = F.avg_pool2d(scale8, kernel_size=2, stride=2)
        scale32 = F.avg_pool2d(scale16, kernel_size=2, stride=2)
        return [scale4, scale8, scale16, scale32]


class SpatialGatedFusion(nn.Module):
    """Geometry-derived spatial gate applied multiplicatively to RGB features."""

    def __init__(self, geometry_channels: int, rgb_channels: int) -> None:
        super().__init__()
        hidden = max(8, min(32, geometry_channels))
        self.geometry_refine = DepthwiseSeparableConv(
            geometry_channels, hidden, kernel_size=3
        )
        self.spatial_gate = nn.Sequential(
            DepthwiseSeparableConv(hidden, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.rgb_refine = DepthwiseSeparableConv(
            rgb_channels, rgb_channels, kernel_size=3
        )

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry_refine(geometry)
        gate = self.spatial_gate(geometry)
        # The [0.5, 1.5] range lets geometry suppress uncertain regions without
        # erasing the RGB subject, matching the recall-first project objective.
        gated = rgb * (0.5 + gate)
        return self.rgb_refine(gated)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(self.pool(x))


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class MultiScaleDepthwiseBlock(nn.Module):
    """RKNN-friendly form of EMCAD's multi-scale depthwise convolution block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch3 = ConvBNAct(channels, channels, 3, groups=channels)
        self.branch5 = ConvBNAct(channels, channels, 5, groups=channels)
        self.branch7 = ConvBNAct(channels, channels, 7, groups=channels)
        self.project = ConvBNAct(channels, channels, 1)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = x + self.branch3(x) + self.branch5(x) + self.branch7(x)
        mixed = self.project(mixed)
        return self.spatial_attention(self.channel_attention(mixed))


class LargeKernelAttentionGate(nn.Module):
    """Light grouped attention gate used when merging an encoder skip."""

    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder_projection = ConvBNAct(decoder_channels, out_channels, 1, activation=False)
        self.skip_projection = ConvBNAct(skip_channels, out_channels, 1, activation=False)
        groups = max(1, out_channels // 8)
        self.attention = nn.Sequential(
            nn.ReLU6(inplace=True),
            ConvBNAct(out_channels, out_channels, 3, groups=groups),
            nn.Conv2d(out_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = self.attention(
            self.decoder_projection(decoder) + self.skip_projection(skip)
        )
        return skip * gate


class EMCADStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_projection = DepthwiseSeparableConv(in_channels, out_channels)
        self.skip_projection = ConvBNAct(skip_channels, out_channels, 1)
        self.gate = LargeKernelAttentionGate(out_channels, out_channels, out_channels)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(2 * out_channels, out_channels),
            MultiScaleDepthwiseBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_projection(self.upsample(x))
        skip = self.skip_projection(skip)
        skip = self.gate(x, skip)
        return self.fuse(torch.cat((x, skip), dim=1))


class EMCADDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (24, 32, 48, 96),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d1, d2, d3, d4 = decoder_channels
        self.bottleneck = nn.Sequential(
            ConvBNAct(c4, d4, 1),
            MultiScaleDepthwiseBlock(d4),
        )
        self.stage3 = EMCADStage(d4, c3, d3)
        self.stage2 = EMCADStage(d3, c2, d2)
        self.stage1 = EMCADStage(d2, c1, d1)
        self.full_resolution = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(d1, d1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        x = self.bottleneck(f4)
        x = self.stage3(x, f3)
        x = self.stage2(x, f2)
        x = self.stage1(x, f1)
        return self.full_resolution(x)


class LightweightRGBDStudent(nn.Module):
    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (24, 32, 48, 96),
    ) -> None:
        super().__init__()
        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        rgb_channels = tuple(self.rgb_encoder.feature_info.channels())
        reductions = tuple(self.rgb_encoder.feature_info.reduction())
        if reductions != (4, 8, 16, 32):
            raise ValueError(f"Unexpected MobileNetV4 reductions: {reductions}")
        self.geometry_encoder = ShallowGeometryEncoder()
        geometry_channels = (24, 32, 32, 32)
        self.fusions = nn.ModuleList(
            SpatialGatedFusion(g_channels, r_channels)
            for g_channels, r_channels in zip(geometry_channels, rgb_channels)
        )
        self.decoder = EMCADDecoder(rgb_channels, decoder_channels)
        head_channels = int(decoder_channels[0])
        self.mask_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    def forward(
        self, rgb: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_features = self.rgb_encoder(rgb)
        geometry_features = self.geometry_encoder(geometry)
        fused = [
            fusion(rgb_feature, geometry_feature)
            for fusion, rgb_feature, geometry_feature in zip(
                self.fusions, rgb_features, geometry_features
            )
        ]
        decoded = self.decoder(fused)
        mask = self.mask_head(decoded)
        boundary = self.boundary_head(decoded)
        return mask, boundary


def create_student(pretrained: bool = True, **kwargs) -> LightweightRGBDStudent:
    return LightweightRGBDStudent(pretrained=pretrained, **kwargs)


if __name__ == "__main__":
    network = create_student(pretrained=False).eval()
    with torch.inference_mode():
        outputs = network(
            torch.randn(1, 3, 1024, 576),
            torch.randn(1, 3, 1024, 576),
        )
    print([tuple(value.shape) for value in outputs])
    print(sum(parameter.numel() for parameter in network.parameters()))
