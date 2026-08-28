"""RGB-only MobileNetV4 + EMCAD foreground segmenter.

The decoder implementation is reused from Experiment 8 so its learned RGB
encoder, decoder and output heads can initialize this pre-stereo model.  The
geometry encoder and geometry gates are deliberately absent: inference only
needs an RGB image and can therefore run before LiteAnyStereo.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import timm
import torch
import torch.nn as nn


def _load_experiment8_network() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "08_lightweight_rgbd/models/student_network.py"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "_jietai_experiment8_student_network"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Experiment 8 network from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LightweightRGBSegmenter(nn.Module):
    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        decoder_channels: tuple[int, int, int, int] = (24, 32, 48, 96),
    ) -> None:
        super().__init__()
        shared = _load_experiment8_network()
        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        channels = tuple(self.rgb_encoder.feature_info.channels())
        reductions = tuple(self.rgb_encoder.feature_info.reduction())
        if reductions != (4, 8, 16, 32):
            raise ValueError(f"Unexpected MobileNetV4 reductions: {reductions}")
        self.decoder = shared.EMCADDecoder(channels, decoder_channels)
        head_channels = int(decoder_channels[0])
        self.mask_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.decoder(self.rgb_encoder(rgb))
        return self.mask_head(decoded), self.boundary_head(decoded)

    def initialize_from_rgbd(self, checkpoint: Path) -> dict[str, int]:
        """Load shape-compatible RGB/decoder/head tensors from Experiment 8."""

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = state.get("model", state)
        target = self.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return {
            "loaded_tensors": len(compatible),
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
        }


def create_rgb_segmenter(
    pretrained: bool = True, **kwargs
) -> LightweightRGBSegmenter:
    return LightweightRGBSegmenter(pretrained=pretrained, **kwargs)


if __name__ == "__main__":
    network = create_rgb_segmenter(pretrained=False).eval()
    with torch.inference_mode():
        outputs = network(torch.randn(1, 3, 1024, 576))
    print([tuple(value.shape) for value in outputs])
    print(sum(parameter.numel() for parameter in network.parameters()))
