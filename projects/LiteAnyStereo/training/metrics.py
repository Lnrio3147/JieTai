"""Streaming and scene-macro disparity metrics used during training and validation."""

from dataclasses import dataclass

import numpy as np
import torch


TRADITION_EXCLUDED_SCENES = (
    "202506281607-0012",
    "202506281609-0019",
    "202506281609-0020",
    "202506281619-0053",
)


@torch.no_grad()
def compute_disparity_metrics(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("prediction, target, and valid must have identical [B,1,H,W] shapes")

    mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    count = int(mask.sum().item())
    if count == 0:
        return {
            "epe_sum": 0.0,
            "d1_count": 0.0,
            "bad1_count": 0.0,
            "bad2_count": 0.0,
            "bad3_count": 0.0,
            "valid_count": 0,
        }

    error = (prediction - target).abs()[mask]
    reference = target.abs()[mask].clamp_min(1e-5)
    d1 = (error > 3.0) & ((error / reference) > 0.05)
    return {
        "epe_sum": float(error.sum().item()),
        "d1_count": float(d1.sum().item()),
        "bad1_count": float((error > 1.0).sum().item()),
        "bad2_count": float((error > 2.0).sum().item()),
        "bad3_count": float((error > 3.0).sum().item()),
        "valid_count": count,
    }


def disparity_metric_row(prediction, target, valid, *, name, total_pixels=None):
    """Return tradition_stereo-compatible metrics for one scene."""
    raw = compute_disparity_metrics(prediction, target, valid)
    count = raw["valid_count"]
    total = int(total_pixels if total_pixels is not None else target.numel())
    if count == 0:
        return {
            "scene": str(name),
            "epe": 0.0,
            "d1": 0.0,
            "bad1": 0.0,
            "bad2": 0.0,
            "bad3": 0.0,
            "valid_pixels": 0,
            "total_pixels": total,
            "valid_ratio": 0.0,
        }
    scale = 100.0 / count
    return {
        "scene": str(name),
        "epe": raw["epe_sum"] / count,
        "d1": raw["d1_count"] * scale,
        "bad1": raw["bad1_count"] * scale,
        "bad2": raw["bad2_count"] * scale,
        "bad3": raw["bad3_count"] * scale,
        "valid_pixels": count,
        "total_pixels": total,
        "valid_ratio": 100.0 * count / total if total else 0.0,
    }


def aggregate_tradition_metrics(scene_rows, *, excluded_scenes=(), epe_threshold=20.0):
    """Apply tradition_stereo scene exclusion/filtering and macro averaging."""
    rows = list(scene_rows)
    excluded_set = set(excluded_scenes)
    excluded = [row for row in rows if row["scene"] in excluded_set]
    candidates = [row for row in rows if row["scene"] not in excluded_set]
    filtered = (
        [row for row in candidates if row["epe"] > epe_threshold]
        if epe_threshold is not None
        else []
    )
    kept = (
        [row for row in candidates if row["epe"] <= epe_threshold]
        if epe_threshold is not None
        else candidates
    )
    if not kept:
        raise ValueError("Tradition evaluation retained no scenes after exclusion/filtering")

    result = {
        key: float(np.mean([row[key] for row in kept]))
        for key in ("epe", "d1", "bad1", "bad2", "bad3", "valid_ratio")
    }
    result.update(
        {
            "valid_pixels": int(sum(row["valid_pixels"] for row in kept)),
            "total_pixels": int(sum(row["total_pixels"] for row in kept)),
            "scene_count": len(kept),
            "original_scene_count": len(rows),
            "excluded_scene_count": len(excluded),
            "epe_filtered_scene_count": len(filtered),
            "excluded_scenes": [row["scene"] for row in excluded],
            "epe_filtered_scenes": [row["scene"] for row in filtered],
            "epe_filter_threshold": epe_threshold,
            "aggregation": "scene_macro",
        }
    )
    return result


@dataclass
class DisparityMetrics:
    epe_sum: float = 0.0
    d1_count: float = 0.0
    bad1_count: float = 0.0
    bad2_count: float = 0.0
    bad3_count: float = 0.0
    valid_count: int = 0

    def update(self, prediction, target, valid):
        batch = compute_disparity_metrics(prediction, target, valid)
        self.epe_sum += batch["epe_sum"]
        self.d1_count += batch["d1_count"]
        self.bad1_count += batch["bad1_count"]
        self.bad2_count += batch["bad2_count"]
        self.bad3_count += batch["bad3_count"]
        self.valid_count += batch["valid_count"]

    def compute(self):
        if self.valid_count == 0:
            return {
                "epe": float("nan"),
                "d1": float("nan"),
                "bad1": float("nan"),
                "bad2": float("nan"),
                "bad3": float("nan"),
                "valid_pixels": 0,
            }
        scale = 100.0 / self.valid_count
        return {
            "epe": self.epe_sum / self.valid_count,
            "d1": self.d1_count * scale,
            "bad1": self.bad1_count * scale,
            "bad2": self.bad2_count * scale,
            "bad3": self.bad3_count * scale,
            "valid_pixels": self.valid_count,
        }
