"""Reusable RGB-only segmentation inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.rgb_segmenter import create_rgb_segmenter
from utils.data import preprocess_rgb


class RGBSegmenterPredictor:
    def __init__(
        self,
        checkpoint: Path,
        width: int,
        height: int,
        device: torch.device,
        no_amp: bool = False,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = create_rgb_segmenter(pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self.amp = device.type == "cuda" and not no_amp
        self.threshold = float(state.get("selected_threshold", 0.5))

    @torch.inference_mode()
    def predict_images(
        self, images_bgr: list[np.ndarray] | tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not images_bgr:
            raise ValueError("At least one image is required")
        shapes = {image.shape for image in images_bgr}
        if len(shapes) != 1:
            raise ValueError(f"Image shape mismatch: {sorted(shapes)}")
        batch = torch.stack(
            [preprocess_rgb(image, self.width, self.height) for image in images_bgr]
        ).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            logits, boundary_logits = self.model(batch)
        probability = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        boundary = torch.sigmoid(boundary_logits).float().cpu().numpy()[:, 0]
        first = images_bgr[0]
        size = (first.shape[1], first.shape[0])
        probability = np.stack(
            [cv2.resize(value, size, interpolation=cv2.INTER_LINEAR) for value in probability]
        ).astype(np.float32)
        boundary = np.stack(
            [cv2.resize(value, size, interpolation=cv2.INTER_LINEAR) for value in boundary]
        ).astype(np.float32)
        return probability, boundary

    def predict_image(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability, boundary = self.predict_images((image_bgr,))
        return probability[0], boundary[0]

    def predict_pair(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if left_bgr.shape != right_bgr.shape:
            raise ValueError(
                f"Left/right image shape mismatch: {left_bgr.shape} vs {right_bgr.shape}"
            )
        probability, boundary = self.predict_images((left_bgr, right_bgr))
        return probability[0], probability[1], boundary[0], boundary[1]
