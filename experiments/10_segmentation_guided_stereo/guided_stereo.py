#!/usr/bin/env python3
"""Run baseline, post-mask, ROI, or soft-mask-guided LiteAnyStereo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config_experiment10 as config
from utils.roi import StereoROI, select_common_roi
from utils.segmentation import RGBSegmenterPredictor


if str(config.LAS_ROOT) not in sys.path:
    sys.path.insert(0, str(config.LAS_ROOT))

from core.models import build_model, load_model_weights  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


METHODS = ("baseline", "post_mask", "roi", "guided")


def load_las(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = build_model("las1", fnet_pretrained=False).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    load_model_weights(model, state, strict=True)
    return model


def bgr_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .float()[None]
        .to(device)
    )


@torch.inference_mode()
def infer_las(
    model: torch.nn.Module,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    device: torch.device,
    max_disparity: int,
    left_probability: np.ndarray | None = None,
    right_probability: np.ndarray | None = None,
    guidance_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    if left_bgr.shape != right_bgr.shape:
        raise ValueError(
            f"Left/right image shape mismatch: {left_bgr.shape} vs {right_bgr.shape}"
        )
    if max_disparity % 4 != 0:
        raise ValueError("max_disparity must be divisible by four")
    left = bgr_tensor(left_bgr, device)
    right = bgr_tensor(right_bgr, device)
    height, width = left_bgr.shape[:2]
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    model_kwargs = {}
    if guidance_weight > 0:
        if left_probability is None or right_probability is None:
            raise ValueError("Both probability maps are required for guidance")
        if left_probability.shape != (height, width):
            raise ValueError(
                f"Left probability shape {left_probability.shape} != {(height, width)}"
            )
        if right_probability.shape != (height, width):
            raise ValueError(
                f"Right probability shape {right_probability.shape} != {(height, width)}"
            )
        left_mask = torch.from_numpy(left_probability.astype(np.float32))[None, None].to(
            device
        )
        right_mask = torch.from_numpy(right_probability.astype(np.float32))[None, None].to(
            device
        )
        padding = padder._pad
        left_mask = F.pad(left_mask, padding, mode="constant", value=0.0)
        right_mask = F.pad(right_mask, padding, mode="constant", value=0.0)
        model_kwargs = {
            "left_mask": left_mask,
            "right_mask": right_mask,
            "mask_guidance_weight": float(guidance_weight),
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(
        left,
        right,
        max_disp=max_disparity,
        test_mode=True,
        **model_kwargs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = padder.unpad(prediction.float())[0, 0].cpu().numpy()
    return prediction.astype(np.float32), float(elapsed)


def run_method(
    method: str,
    model: torch.nn.Module,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    device: torch.device,
    *,
    threshold: float,
    max_disparity: int,
    guidance_weight: float,
    roi_margin: int,
) -> tuple[np.ndarray, np.ndarray, StereoROI, dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}")
    height, width = left_bgr.shape[:2]
    full_roi = StereoROI(
        0,
        0,
        width,
        height,
        False,
        "method_uses_full_frame",
        1.0,
        int((left_probability >= threshold).sum()),
        int((right_probability >= threshold).sum()),
    )
    roi = full_roi
    use_roi = method in ("roi", "guided")
    if use_roi:
        roi = select_common_roi(
            left_probability,
            right_probability,
            threshold=threshold,
            margin=roi_margin,
            max_disparity=max_disparity,
            stride=config.ROI_STRIDE,
            min_foreground_pixels=config.ROI_MIN_FOREGROUND_PIXELS,
            min_area_ratio=config.ROI_MIN_AREA_RATIO,
            max_area_ratio=config.ROI_MAX_AREA_RATIO,
        )
    y_slice, x_slice = roi.slices()
    crop_left = left_bgr[y_slice, x_slice]
    crop_right = right_bgr[y_slice, x_slice]
    crop_left_probability = left_probability[y_slice, x_slice]
    crop_right_probability = right_probability[y_slice, x_slice]
    active_guidance = guidance_weight if method == "guided" else 0.0
    crop_disparity, stereo_seconds = infer_las(
        model,
        crop_left,
        crop_right,
        device,
        max_disparity,
        crop_left_probability,
        crop_right_probability,
        active_guidance,
    )
    disparity = np.full((height, width), np.nan, dtype=np.float32)
    disparity[y_slice, x_slice] = crop_disparity
    if method in ("baseline", "post_mask"):
        # The full ROI above makes this assignment cover the complete image.
        assert np.isfinite(disparity).all()
    subject = disparity.copy()
    subject[left_probability < threshold] = np.nan
    diagnostics = correspondence_diagnostics(
        disparity,
        left_probability,
        right_probability,
        left_bgr,
        right_bgr,
        threshold,
    )
    diagnostics.update(
        {
            "method": method,
            "stereo_seconds": stereo_seconds,
            "guidance_weight": active_guidance,
            "roi": roi.to_dict(),
            "computed_pixel_fraction": float(roi.area_ratio),
        }
    )
    return disparity, subject, roi, diagnostics


def sample_right_probability(
    right_probability: np.ndarray, disparity: np.ndarray
) -> np.ndarray:
    height, width = disparity.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx - np.nan_to_num(disparity, nan=0.0).astype(np.float32)
    return cv2.remap(
        right_probability.astype(np.float32),
        map_x,
        yy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def correspondence_diagnostics(
    disparity: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    threshold: float,
) -> dict:
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & (left_probability >= threshold)
    )
    support = sample_right_probability(right_probability, disparity)
    violation = valid & (support < threshold)
    height, width = disparity.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx - np.nan_to_num(disparity, nan=0.0).astype(np.float32)
    warped_right = cv2.remap(
        cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY),
        map_x,
        yy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    photometric = np.abs(left_gray.astype(np.float32) - warped_right.astype(np.float32))
    return {
        "valid_subject_pixels": int(valid.sum()),
        "right_mask_violation_pixels": int(violation.sum()),
        "right_mask_violation_rate": float(violation.sum() / max(valid.sum(), 1)),
        "mean_right_mask_support": float(support[valid].mean()) if np.any(valid) else None,
        "mean_subject_photometric_error": (
            float(photometric[valid].mean()) if np.any(valid) else None
        ),
    }


def load_q(path: Path) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)
    value = storage.getNode("Q").mat()
    storage.release()
    if value is None or value.shape != (4, 4):
        raise ValueError(f"Invalid Q matrix in {path}")
    return value.astype(np.float32)


def save_subject_cloud(
    path: Path,
    subject_disparity: np.ndarray,
    left_bgr: np.ndarray,
    q: np.ndarray,
    max_abs_z: float | None,
) -> int:
    import open3d as o3d

    safe = np.nan_to_num(subject_disparity, nan=0.0).astype(np.float32)
    xyz = cv2.reprojectImageTo3D(safe, q, handleMissingValues=False)
    valid = np.isfinite(subject_disparity) & (subject_disparity > 0)
    valid &= np.isfinite(xyz).all(axis=2)
    if max_abs_z is not None:
        valid &= np.abs(xyz[..., 2]) <= max_abs_z
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz[valid].astype(np.float64))
    rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise IOError(f"Failed to write {path}")
    return len(cloud.points)


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(disparity[valid], [2.0, 98.0])
        scale = max(float(high - low), 1e-6)
        normalized[valid] = np.clip(
            (disparity[valid] - low) * 255.0 / scale, 0, 255
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    # Frozen FDJYP-3 testing showed that the soft prior improves mask and
    # photometric consistency but can worsen engineering-reference EPE.
    # Keep the deploy-facing single-pair entry point on the safe post-mask path.
    parser.add_argument("--method", choices=METHODS, default="post_mask")
    parser.add_argument("--segmenter", type=Path, default=config.SEGMENTER_CHECKPOINT)
    parser.add_argument("--las-checkpoint", type=Path, default=config.LAS_CHECKPOINT)
    parser.add_argument("--left-probability", type=Path, default=None)
    parser.add_argument("--right-probability", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--guidance-weight", type=float, default=config.MASK_GUIDANCE_WEIGHT)
    parser.add_argument("--roi-margin", type=int, default=config.ROI_MARGIN)
    parser.add_argument("--max-disparity", type=int, default=config.LAS_MAX_DISPARITY)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--q-matrix", type=Path, default=None)
    parser.add_argument("--max-abs-z", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(args.left)
    if right is None:
        raise FileNotFoundError(args.right)
    predictor = None
    if args.left_probability is not None or args.right_probability is not None:
        if args.left_probability is None or args.right_probability is None:
            raise ValueError("Pass both --left-probability and --right-probability")
        left_probability = np.load(args.left_probability).astype(np.float32)
        right_probability = np.load(args.right_probability).astype(np.float32)
        threshold = config.MASK_THRESHOLD if args.threshold is None else args.threshold
    else:
        predictor = RGBSegmenterPredictor(
            args.segmenter,
            config.IMAGE_WIDTH,
            config.IMAGE_HEIGHT,
            device,
            args.no_amp,
        )
        left_probability, right_probability, _, _ = predictor.predict_pair(left, right)
        threshold = predictor.threshold if args.threshold is None else args.threshold
    model = load_las(args.las_checkpoint, device)
    disparity, subject, _, diagnostics = run_method(
        args.method,
        model,
        left,
        right,
        left_probability,
        right_probability,
        device,
        threshold=threshold,
        max_disparity=args.max_disparity,
        guidance_weight=args.guidance_weight,
        roi_margin=args.roi_margin,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "disparity.npy", disparity)
    np.save(args.output / "subject_disparity.npy", subject)
    np.save(args.output / "left_probability.npy", left_probability.astype(np.float16))
    np.save(args.output / "right_probability.npy", right_probability.astype(np.float16))
    cv2.imwrite(str(args.output / "disparity.png"), colorize_disparity(disparity))
    cv2.imwrite(str(args.output / "subject_disparity.png"), colorize_disparity(subject))
    cv2.imwrite(
        str(args.output / "left_mask.png"),
        (left_probability >= threshold).astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(args.output / "right_mask.png"),
        (right_probability >= threshold).astype(np.uint8) * 255,
    )
    if args.q_matrix is not None:
        diagnostics["point_count"] = save_subject_cloud(
            args.output / "subject_cloud.ply",
            subject,
            left,
            load_q(args.q_matrix),
            args.max_abs_z,
        )
    diagnostics.update(
        {
            "left": str(args.left.resolve()),
            "right": str(args.right.resolve()),
            "segmenter": str(args.segmenter.resolve()) if predictor is not None else None,
            "las_checkpoint": str(args.las_checkpoint.resolve()),
            "mask_threshold": float(threshold),
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
