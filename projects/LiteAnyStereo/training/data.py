"""Dataset adapters for public, manifest-based, and synthetic stereo data."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from core.utils.frame_utils import readPFM


TRADITION_CROP = (234, 1052, 126, 638)


@dataclass(frozen=True)
class StereoSample:
    left: Path
    right: Path
    disparity: Path
    valid: Path | None = None
    name: str = ""
    disparity_scale: float = 1.0


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def read_disparity(path: Path, scale: float = 1.0) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Disparity not found: {path}")
    if scale <= 0:
        raise ValueError(f"Disparity scale must be positive, got {scale}")

    suffix = path.suffix.lower()
    if suffix == ".pfm":
        disparity = readPFM(str(path))
    elif suffix in {".npy", ".bin", ".raw"}:
        disparity = np.load(path)
    elif suffix in {".png", ".tif", ".tiff"}:
        disparity = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise ValueError(f"OpenCV could not read disparity: {path}")
    else:
        raise ValueError(f"Unsupported disparity format '{suffix}' for {path}")

    disparity = np.asarray(disparity)
    if disparity.ndim == 3 and disparity.shape[-1] == 1:
        disparity = disparity[..., 0]
    if disparity.ndim != 2:
        raise ValueError(f"Expected a single-channel disparity map, got {disparity.shape} from {path}")
    return disparity.astype(np.float32) / float(scale)


def read_valid_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Validity mask not found: {path}")
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"OpenCV could not read validity mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel validity mask, got {mask.shape} from {path}")
    return np.asarray(mask) > 0


def load_manifest_samples(manifest: str | Path, split: str) -> list[StereoSample]:
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"left", "right", "disparity", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        for row_index, row in enumerate(reader, start=2):
            if row["split"].strip().lower() != split.lower():
                continue
            try:
                scale = float(row.get("disp_scale") or 1.0)
            except ValueError as error:
                raise ValueError(f"Invalid disp_scale on manifest row {row_index}") from error
            valid_value = (row.get("valid") or "").strip()
            name = (row.get("name") or "").strip() or Path(row["left"]).stem
            samples.append(
                StereoSample(
                    left=_resolve_path(manifest_path.parent, row["left"].strip()),
                    right=_resolve_path(manifest_path.parent, row["right"].strip()),
                    disparity=_resolve_path(manifest_path.parent, row["disparity"].strip()),
                    valid=_resolve_path(manifest_path.parent, valid_value) if valid_value else None,
                    name=name,
                    disparity_scale=scale,
                )
            )

    if not samples:
        raise ValueError(f"Manifest {manifest_path} contains no rows for split '{split}'.")
    return samples


def _pad_to_size(image, height, width, *, is_mask=False):
    pad_height = max(height - image.shape[0], 0)
    pad_width = max(width - image.shape[1], 0)
    if pad_height == 0 and pad_width == 0:
        return image
    if image.ndim == 3:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        padding = ((0, pad_height), (0, pad_width))
    if is_mask:
        return np.pad(image, padding, mode="constant", constant_values=0)
    return np.pad(image, padding, mode="edge")


def _random_crop(left, right, disparity, valid, crop_height, crop_width):
    left = _pad_to_size(left, crop_height, crop_width)
    right = _pad_to_size(right, crop_height, crop_width)
    disparity = _pad_to_size(disparity, crop_height, crop_width, is_mask=True)
    valid = _pad_to_size(valid, crop_height, crop_width, is_mask=True)

    height, width = disparity.shape
    max_y = height - crop_height
    max_x = width - crop_width
    best = None
    best_count = -1
    for _ in range(10):
        y = random.randint(0, max_y) if max_y else 0
        x = random.randint(0, max_x) if max_x else 0
        count = int(valid[y : y + crop_height, x : x + crop_width].sum())
        if count > best_count:
            best = (y, x)
            best_count = count
        if count >= crop_height * crop_width * 0.05:
            break

    y, x = best
    region = np.s_[y : y + crop_height, x : x + crop_width]
    return left[region], right[region], disparity[region], valid[region]


class FileStereoDataset(Dataset):
    def __init__(self, samples, max_disp=192, crop_size=None, training=False):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Stereo dataset cannot be empty.")
        self.max_disp = float(max_disp)
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.training = bool(training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        left = read_rgb(sample.left)
        right = read_rgb(sample.right)
        disparity = read_disparity(sample.disparity, sample.disparity_scale)

        if left.shape != right.shape:
            raise ValueError(f"Left/right image shape mismatch for {sample.name}: {left.shape} vs {right.shape}")
        if left.shape[:2] != disparity.shape:
            raise ValueError(
                f"Image/disparity shape mismatch for {sample.name}: {left.shape[:2]} vs {disparity.shape}"
            )

        if sample.valid is None:
            valid = np.ones(disparity.shape, dtype=bool)
        else:
            valid = read_valid_mask(sample.valid)
            if valid.shape != disparity.shape:
                raise ValueError(
                    f"Disparity/mask shape mismatch for {sample.name}: {disparity.shape} vs {valid.shape}"
                )
        valid &= np.isfinite(disparity) & (disparity > 0.0) & (disparity < self.max_disp)

        if self.training and self.crop_size is not None:
            left, right, disparity, valid = _random_crop(
                left,
                right,
                disparity,
                valid,
                self.crop_size[0],
                self.crop_size[1],
            )

        return {
            "left": torch.from_numpy(np.ascontiguousarray(left)).permute(2, 0, 1).float(),
            "right": torch.from_numpy(np.ascontiguousarray(right)).permute(2, 0, 1).float(),
            "disparity": torch.from_numpy(np.ascontiguousarray(disparity))[None].float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid))[None].bool(),
            "name": sample.name,
        }


class ManifestStereoDataset(FileStereoDataset):
    def __init__(self, manifest, split, max_disp=192, crop_size=None, training=False):
        self.manifest = str(Path(manifest).expanduser().resolve())
        self.split = split
        super().__init__(
            load_manifest_samples(self.manifest, split),
            max_disp=max_disp,
            crop_size=crop_size,
            training=training,
        )


class KITTIStereo2015Dataset(FileStereoDataset):
    """Deterministic local split of KITTI 2015's 200 labeled training scenes."""

    def __init__(
        self,
        root,
        split,
        max_disp=192,
        crop_size=None,
        training=False,
        val_fraction=0.2,
        split_seed=42,
    ):
        root = Path(root).expanduser().resolve()
        training_root = root / "training"
        left_files = sorted((training_root / "image_2").glob("*_10.png"))
        if not left_files:
            raise FileNotFoundError(
                f"No KITTI 2015 training images found under {training_root / 'image_2'}. "
                "Expected files such as 000000_10.png."
            )

        all_samples = []
        for left in left_files:
            right = training_root / "image_3" / left.name
            disparity = training_root / "disp_occ_0" / left.name
            all_samples.append(
                StereoSample(
                    left=left,
                    right=right,
                    disparity=disparity,
                    name=left.stem,
                    disparity_scale=256.0,
                )
            )

        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        generator = np.random.default_rng(split_seed)
        indices = generator.permutation(len(all_samples))
        val_count = max(1, int(round(len(all_samples) * val_fraction)))
        val_indices = set(int(index) for index in indices[:val_count])
        if split == "train":
            samples = [sample for index, sample in enumerate(all_samples) if index not in val_indices]
        elif split == "val":
            samples = [sample for index, sample in enumerate(all_samples) if index in val_indices]
        else:
            raise ValueError("KITTI split must be 'train' or 'val'")

        self.root = str(root)
        self.split = split
        self.split_seed = split_seed
        self.val_fraction = val_fraction
        super().__init__(samples, max_disp=max_disp, crop_size=crop_size, training=training)


class ETH3DStereoDataset(FileStereoDataset):
    """Deterministic split of the official ETH3D low-res two-view training set."""

    def __init__(
        self,
        root,
        split,
        max_disp=192,
        crop_size=None,
        training=False,
        val_fraction=0.2,
        split_seed=42,
    ):
        root = Path(root).expanduser().resolve()
        image_root = root / "two_view_training" if (root / "two_view_training").is_dir() else root
        gt_root = root / "two_view_training_gt" if (root / "two_view_training_gt").is_dir() else root
        left_files = sorted(image_root.glob("*/im0.png"))
        if not left_files:
            raise FileNotFoundError(
                f"No ETH3D scenes found under {image_root}. Expected scene folders containing im0.png."
            )

        all_samples = []
        for left in left_files:
            scene = left.parent.name
            ground_truth_dir = gt_root / scene
            valid_path = ground_truth_dir / "mask0nocc.png"
            all_samples.append(
                StereoSample(
                    left=left,
                    right=left.parent / "im1.png",
                    disparity=ground_truth_dir / "disp0GT.pfm",
                    valid=valid_path if valid_path.is_file() else None,
                    name=scene,
                )
            )

        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        # Keep the short/long-baseline variants of one capture in the same split.
        grouped = {}
        for sample in all_samples:
            group_name = sample.name[:-1] if sample.name.endswith(("l", "s")) else sample.name
            grouped.setdefault(group_name, []).append(sample)
        group_names = sorted(grouped)
        generator = np.random.default_rng(split_seed)
        indices = generator.permutation(len(group_names))
        val_count = max(1, int(round(len(group_names) * val_fraction)))
        val_groups = {group_names[int(index)] for index in indices[:val_count]}
        if split == "train":
            samples = [sample for name, group in grouped.items() if name not in val_groups for sample in group]
        elif split == "val":
            samples = [sample for name, group in grouped.items() if name in val_groups for sample in group]
        else:
            raise ValueError("ETH3D split must be 'train' or 'val'")

        self.root = str(root)
        self.split = split
        self.split_seed = split_seed
        self.val_fraction = val_fraction
        super().__init__(samples, max_disp=max_disp, crop_size=crop_size, training=training)


class JMPLF6020Dataset(FileStereoDataset):
    """JMP-LF6020 in ETH3D layout, using explicit capture-group splits."""

    def __init__(self, root, split, max_disp=192, crop_size=None, training=False):
        root = Path(root).expanduser().resolve()
        if split not in {"train", "val"}:
            raise ValueError("JMP-LF6020 split must be 'train' or 'val'")
        split_path = root / "splits" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"JMP-LF6020 split file not found: {split_path}. "
                "Prepare the archive with tools/prepare_jmp_lf6020.py."
            )

        scene_names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(scene_names) != len(set(scene_names)):
            raise ValueError(f"Duplicate scene names found in {split_path}")

        samples = []
        for scene_name in scene_names:
            if Path(scene_name).name != scene_name:
                raise ValueError(f"Invalid scene name in {split_path}: {scene_name!r}")
            scene_dir = root / scene_name
            samples.append(
                StereoSample(
                    left=scene_dir / "im0.png",
                    right=scene_dir / "im1.png",
                    disparity=scene_dir / "disp0GT.pfm",
                    valid=scene_dir / "mask0nocc.png",
                    name=scene_name,
                )
            )

        self.root = str(root)
        self.split = split
        super().__init__(samples, max_disp=max_disp, crop_size=crop_size, training=training)


class TraditionStereoEvaluationDataset(Dataset):
    """Rectified JMP pairs with fixed-ROI tradition_stereo reference disparity."""

    def __init__(self, root, gt_filename="disp_cropped.npy", image_root=None):
        self.root = str(Path(root).expanduser().resolve())
        self.gt_filename = gt_filename
        self.image_root = str(Path(image_root).expanduser().resolve()) if image_root else None
        root_path = Path(self.root)
        reference_scenes = sorted(
            scene
            for scene in root_path.iterdir()
            if scene.is_dir()
            and scene.name.startswith("20")
            and (scene / "im0.png").is_file()
            and (scene / "im1.png").is_file()
            and (scene / gt_filename).is_file()
        )
        if not reference_scenes:
            raise FileNotFoundError(
                f"No tradition_stereo evaluation scenes found under {root_path}; expected "
                f"<scene>/im0.png, im1.png, and {gt_filename}."
            )
        if image_root is None:
            self.scenes = [(scene, scene) for scene in reference_scenes]
        else:
            image_root_path = Path(self.image_root)
            image_scenes = {}
            for scene in image_root_path.iterdir():
                if not scene.is_dir() or not (scene / "im0.png").is_file() or not (scene / "im1.png").is_file():
                    continue
                parts = scene.name.rsplit("_", 2)
                reference_name = f"{parts[-2]}-{parts[-1]}" if len(parts) >= 3 else scene.name
                if reference_name in image_scenes:
                    raise ValueError(f"Duplicate rectified image scene for {reference_name}")
                image_scenes[reference_name] = scene
            missing = [scene.name for scene in reference_scenes if scene.name not in image_scenes]
            if missing:
                raise FileNotFoundError(
                    f"Missing rectified images for {len(missing)} tradition scenes under "
                    f"{image_root_path}; first missing scene: {missing[0]}"
                )
            self.scenes = [(scene, image_scenes[scene.name]) for scene in reference_scenes]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        reference_scene, image_scene = self.scenes[index]
        left = read_rgb(image_scene / "im0.png")
        right = read_rgb(image_scene / "im1.png")
        gt = read_disparity(reference_scene / self.gt_filename)
        scene_number = reference_scene.name.rsplit("-", 1)[-1]
        # The phase-I project baseline was RT-IGEV from the IGEV-plusplus
        # repository.  Prefer its saved prediction; retain the historical
        # customer point-cloud projection only as a compatibility fallback.
        igev_path = reference_scene / "disp_igev.npy"
        historical_path = reference_scene / f"{scene_number}_disp_cropped.npy"
        previous_path = igev_path if igev_path.is_file() else historical_path
        previous_label = "RT-IGEV" if previous_path == igev_path else "Historical result"
        if left.shape != right.shape:
            raise ValueError(f"Left/right image shape mismatch for {reference_scene.name}")

        y0, y1, x0, x1 = TRADITION_CROP
        expected_shape = (y1 - y0, x1 - x0)
        if gt.shape != expected_shape:
            raise ValueError(
                f"Tradition GT shape mismatch for {reference_scene.name}: {gt.shape}, expected {expected_shape}"
            )
        if y1 > left.shape[0] or x1 > left.shape[1]:
            raise ValueError(
                f"Tradition crop exceeds image bounds for {reference_scene.name}: {left.shape[:2]}"
            )

        disparity = np.zeros(left.shape[:2], dtype=np.float32)
        valid = np.zeros(left.shape[:2], dtype=bool)
        disparity[y0:y1, x0:x1] = gt
        valid[y0:y1, x0:x1] = np.isfinite(gt) & (gt > 0.0)
        sample = {
            "left": torch.from_numpy(np.ascontiguousarray(left)).permute(2, 0, 1).float(),
            "right": torch.from_numpy(np.ascontiguousarray(right)).permute(2, 0, 1).float(),
            "disparity": torch.from_numpy(disparity)[None],
            "valid": torch.from_numpy(valid)[None],
            "name": reference_scene.name,
            "evaluation_pixels": int(gt.size),
        }
        if previous_path.is_file():
            traditional = read_disparity(previous_path)
            if traditional.shape != expected_shape:
                raise ValueError(
                    f"Traditional disparity shape mismatch for {reference_scene.name}: "
                    f"{traditional.shape}, expected {expected_shape}"
                )
            traditional_full = np.zeros(left.shape[:2], dtype=np.float32)
            traditional_full[y0:y1, x0:x1] = traditional
            sample["traditional_disparity"] = torch.from_numpy(traditional_full)[None]
            sample["traditional_label"] = previous_label
        return sample


class SyntheticShiftDataset(Dataset):
    """Small deterministic stereo dataset for checking the training pipeline only."""

    def __init__(self, length=16, height=64, width=256, max_disp=192, seed=42):
        if length <= 0:
            raise ValueError("Synthetic dataset length must be positive")
        if height % 32 or width % 32:
            raise ValueError("Synthetic image height and width must be divisible by 32")
        self.length = int(length)
        self.height = int(height)
        self.width = int(width)
        self.max_disp = int(max_disp)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        base = rng.integers(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        # Add broad structures so the pair is not only uncorrelated pixel noise.
        for _ in range(12):
            y0 = int(rng.integers(0, max(self.height - 4, 1)))
            x0 = int(rng.integers(0, max(self.width - 8, 1)))
            y1 = min(self.height, y0 + int(rng.integers(4, max(self.height // 2, 5))))
            x1 = min(self.width, x0 + int(rng.integers(8, max(self.width // 3, 9))))
            base[y0:y1, x0:x1] = rng.integers(0, 256, size=(1, 1, 3), dtype=np.uint8)

        max_shift = max(2, min(self.max_disp - 1, self.width // 8, 32))
        shift = int(rng.integers(1, max_shift + 1))
        right = np.zeros_like(base)
        right[:, : self.width - shift] = base[:, shift:]
        right[:, self.width - shift :] = base[:, -1:]

        disparity = np.full((self.height, self.width), shift, dtype=np.float32)
        valid = np.zeros((self.height, self.width), dtype=bool)
        valid[:, shift:] = True
        return {
            "left": torch.from_numpy(base).permute(2, 0, 1).float(),
            "right": torch.from_numpy(right).permute(2, 0, 1).float(),
            "disparity": torch.from_numpy(disparity)[None],
            "valid": torch.from_numpy(valid)[None],
            "name": f"synthetic_{index:06d}",
        }


def build_datasets(args):
    crop_size = (args.crop_height, args.crop_width)
    if args.dataset == "synthetic":
        train_dataset = SyntheticShiftDataset(
            length=args.synthetic_train_samples,
            height=args.crop_height,
            width=args.crop_width,
            max_disp=args.max_disp,
            seed=args.seed,
        )
        val_dataset = SyntheticShiftDataset(
            length=args.synthetic_val_samples,
            height=args.crop_height,
            width=args.crop_width,
            max_disp=args.max_disp,
            seed=args.seed + 100000,
        )
    elif args.dataset in {"kitti15", "eth3d"}:
        if not args.data_root:
            raise ValueError(f"--data_root is required for dataset={args.dataset}")
        common = dict(
            root=args.data_root,
            max_disp=args.max_disp,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )
        dataset_class = KITTIStereo2015Dataset if args.dataset == "kitti15" else ETH3DStereoDataset
        train_dataset = dataset_class(
            split="train", crop_size=crop_size, training=True, **common
        )
        val_dataset = dataset_class(split="val", crop_size=None, training=False, **common)
    elif args.dataset == "jmp":
        if not args.data_root:
            raise ValueError("--data_root is required for dataset=jmp")
        common = dict(root=args.data_root, max_disp=args.max_disp)
        train_dataset = JMPLF6020Dataset(
            split="train", crop_size=crop_size, training=True, **common
        )
        if getattr(args, "evaluation_protocol", "standard") == "tradition":
            if not getattr(args, "tradition_eval_root", None):
                raise ValueError("--tradition_eval_root is required for tradition evaluation")
            val_dataset = TraditionStereoEvaluationDataset(
                args.tradition_eval_root, image_root=args.data_root
            )
        else:
            val_dataset = JMPLF6020Dataset(split="val", crop_size=None, training=False, **common)
    elif args.dataset == "manifest":
        if not args.manifest:
            raise ValueError("--manifest is required for dataset=manifest")
        train_dataset = ManifestStereoDataset(
            args.manifest,
            "train",
            max_disp=args.max_disp,
            crop_size=crop_size,
            training=True,
        )
        val_dataset = ManifestStereoDataset(
            args.manifest,
            "val",
            max_disp=args.max_disp,
            crop_size=None,
            training=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_dataset, val_dataset
