import csv
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.utils.frame_utils import writePFM
from training.checkpoint import safe_torch_load, save_training_checkpoint
from training.data import (
    JMPLF6020Dataset,
    ManifestStereoDataset,
    SyntheticShiftDataset,
    TRADITION_CROP,
    TraditionStereoEvaluationDataset,
)
from training.losses import multi_prediction_smooth_l1
from training.engine import validate
from training.metrics import (
    DisparityMetrics,
    aggregate_tradition_metrics,
    compute_disparity_metrics,
)
from training.visualization import (
    save_algorithm_comparison_vis,
    save_inference_vis,
    save_validation_vis,
)


class TrainingUtilitiesTest(unittest.TestCase):
    def test_algorithm_comparison_visualization_writes_six_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            image = torch.zeros((3, 64, 96))
            prediction = torch.full((1, 64, 96), 20.0)
            traditional = torch.full((1, 64, 96), 15.0)
            output = Path(directory) / "traditional_comparison.png"
            save_algorithm_comparison_vis(
                output,
                left=image,
                traditional=traditional,
                prediction=prediction,
            )
            with Image.open(output) as comparison:
                self.assertEqual(comparison.size, (288, 204))

    def test_inference_visualization_writes_prediction_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            image = torch.zeros((3, 64, 96))
            prediction = torch.full((1, 64, 96), 20.0)
            output = save_inference_vis(
                directory,
                left=image,
                right=image,
                prediction=prediction,
            )
            with Image.open(output / "vis.png") as visual:
                self.assertEqual(visual.size, (96, 64))
            with Image.open(output / "vis_fixed.png") as fixed:
                self.assertEqual(fixed.size, (96, 64))
            with Image.open(output / "comparison.png") as comparison:
                self.assertEqual(comparison.size, (192, 204))

    def test_validation_visualization_writes_tradition_montage(self):
        with tempfile.TemporaryDirectory() as directory:
            left = torch.zeros((3, 1280, 720))
            prediction = torch.full((1, 1280, 720), 20.0)
            target = torch.full((1, 1280, 720), 10.0)
            traditional = torch.full((1, 1280, 720), 15.0)
            valid = torch.zeros((1, 1280, 720), dtype=torch.bool)
            y0, y1, x0, x1 = TRADITION_CROP
            valid[:, y0:y1, x0:x1] = True
            output = Path(directory) / "scene" / "vis.png"
            save_validation_vis(
                output,
                left=left,
                prediction=prediction,
                target=target,
                valid=valid,
                evaluation_protocol="tradition",
                traditional=traditional,
            )
            with Image.open(output) as montage:
                self.assertEqual(montage.size, (512, 818))
            with Image.open(output.with_name("vis_fixed.png")) as fixed:
                self.assertEqual(fixed.size, (512, 818))
            with Image.open(output.with_name("comparison.png")) as comparison:
                self.assertEqual(comparison.size, (1024, 1712))
            with Image.open(output.with_name("traditional_comparison.png")) as comparison:
                self.assertEqual(comparison.size, (1536, 1712))
            saved_disparity = np.load(output.with_name("disp.npy"))
            self.assertEqual(saved_disparity.shape, (818, 512))
            self.assertTrue(np.allclose(saved_disparity, 20.0))

    def test_tradition_dataset_places_fixed_crop_reference_in_full_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "202506281603-0001"
            scene.mkdir()
            image = np.zeros((1280, 720, 3), dtype=np.uint8)
            Image.fromarray(image).save(scene / "im0.png")
            Image.fromarray(image).save(scene / "im1.png")
            gt = np.full((818, 512), 12.5, dtype=np.float32)
            np.save(scene / "disp_cropped.npy", gt)
            traditional = np.full((818, 512), 10.0, dtype=np.float32)
            np.save(scene / "0001_disp_cropped.npy", traditional)
            igev = np.full((818, 512), 11.0, dtype=np.float32)
            np.save(scene / "disp_igev.npy", igev)

            sample = TraditionStereoEvaluationDataset(root)[0]
            y0, y1, x0, x1 = TRADITION_CROP
            self.assertEqual(tuple(sample["disparity"].shape), (1, 1280, 720))
            self.assertEqual(int(sample["valid"].sum()), gt.size)
            self.assertAlmostEqual(
                float(sample["disparity"][0, y0:y1, x0:x1].median()), 12.5
            )
            self.assertEqual(sample["evaluation_pixels"], gt.size)
            self.assertAlmostEqual(
                float(sample["traditional_disparity"][0, y0:y1, x0:x1].median()), 11.0
            )
            self.assertEqual(sample["traditional_label"], "RT-IGEV")

    def test_tradition_dataset_can_pair_rectified_images_by_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "references" / "202506281603-0001"
            rectified = root / "images" / "fdjyp_3_1_202506281603_0001"
            reference.mkdir(parents=True)
            rectified.mkdir(parents=True)
            Image.fromarray(np.zeros((1280, 720, 3), dtype=np.uint8)).save(
                reference / "im0.png"
            )
            Image.fromarray(np.zeros((1280, 720, 3), dtype=np.uint8)).save(
                reference / "im1.png"
            )
            Image.fromarray(np.full((1280, 720, 3), 17, dtype=np.uint8)).save(
                rectified / "im0.png"
            )
            Image.fromarray(np.full((1280, 720, 3), 23, dtype=np.uint8)).save(
                rectified / "im1.png"
            )
            np.save(reference / "disp_cropped.npy", np.ones((818, 512), np.float32))

            sample = TraditionStereoEvaluationDataset(
                root / "references", image_root=root / "images"
            )[0]
            self.assertAlmostEqual(float(sample["left"].mean()), 17.0)
            self.assertAlmostEqual(float(sample["right"].mean()), 23.0)
            self.assertEqual(sample["name"], "202506281603-0001")

    def test_jmp_eth3d_layout_uses_explicit_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "splits").mkdir()
            image = np.zeros((32, 64, 3), dtype=np.uint8)
            disparity = np.full((32, 64), 7.5, dtype=np.float32)
            for split, scene in (("train", "capture_a"), ("val", "capture_b")):
                scene_dir = root / scene
                scene_dir.mkdir()
                Image.fromarray(image).save(scene_dir / "im0.png")
                Image.fromarray(image).save(scene_dir / "im1.png")
                Image.fromarray(np.ones((32, 64), dtype=np.uint8) * 255).save(
                    scene_dir / "mask0nocc.png"
                )
                writePFM(str(scene_dir / "disp0GT.pfm"), disparity)
                (root / "splits" / f"{split}.txt").write_text(scene + "\n", encoding="utf-8")

            train = JMPLF6020Dataset(root, "train", crop_size=(32, 64), training=True)
            val = JMPLF6020Dataset(root, "val", training=False)
            self.assertEqual(train.samples[0].name, "capture_a")
            self.assertEqual(val.samples[0].name, "capture_b")
            self.assertAlmostEqual(float(train[0]["disparity"].median()), 7.5)

    def test_loss_supports_final_and_auxiliary_predictions(self):
        target = torch.full((1, 1, 2, 3), 4.0)
        valid = torch.ones_like(target, dtype=torch.bool)
        final = torch.full_like(target, 5.0, requires_grad=True)
        auxiliary = torch.full_like(target, 6.0, requires_grad=True)
        loss, components, count = multi_prediction_smooth_l1(
            [final, auxiliary], target, valid, aux_weight=0.5
        )
        self.assertEqual(count, 6)
        self.assertEqual(len(components), 2)
        self.assertAlmostEqual(float(loss), 1.25)
        loss.backward()
        self.assertIsNotNone(final.grad)
        self.assertIsNotNone(auxiliary.grad)

    def test_metrics_match_epe_and_kitti_d1_definition(self):
        prediction = torch.tensor([[[[2.0, 8.0]]]])
        target = torch.tensor([[[[1.0, 4.0]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        raw = compute_disparity_metrics(prediction, target, valid)
        self.assertEqual(raw["valid_count"], 2)
        metrics = DisparityMetrics()
        metrics.update(prediction, target, valid)
        result = metrics.compute()
        self.assertAlmostEqual(result["epe"], 2.5)
        self.assertAlmostEqual(result["d1"], 50.0)
        self.assertAlmostEqual(result["bad1"], 50.0)
        self.assertAlmostEqual(result["bad2"], 50.0)
        self.assertAlmostEqual(result["bad3"], 50.0)

    def test_tradition_macro_average_applies_exclusion_and_epe_filter(self):
        def row(scene, epe):
            return {
                "scene": scene,
                "epe": epe,
                "d1": epe + 1,
                "bad1": epe + 2,
                "bad2": epe + 3,
                "bad3": epe + 4,
                "valid_pixels": 10,
                "total_pixels": 20,
                "valid_ratio": 50.0,
            }

        result = aggregate_tradition_metrics(
            [row("kept", 2.0), row("excluded", 3.0), row("high", 25.0)],
            excluded_scenes=("excluded",),
            epe_threshold=20.0,
        )
        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(result["excluded_scenes"], ["excluded"])
        self.assertEqual(result["epe_filtered_scenes"], ["high"])
        self.assertAlmostEqual(result["epe"], 2.0)
        self.assertAlmostEqual(result["bad3"], 6.0)

    def test_tradition_validation_keeps_gt_above_model_max_disp(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, left, right, max_disp, test_mode):
                del right, max_disp, test_mode
                return torch.zeros(
                    (left.shape[0], 1, left.shape[2], left.shape[3]),
                    dtype=left.dtype,
                    device=left.device,
                )

        target = torch.zeros((1, 1, 32, 32))
        target[0, 0, 0, 0] = 10.0
        target[0, 0, 0, 1] = 200.0
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[0, 0, 0, :2] = True
        batch = {
            "left": torch.zeros((1, 3, 32, 32)),
            "right": torch.zeros((1, 3, 32, 32)),
            "disparity": target,
            "valid": valid,
            "name": ["scene"],
            "evaluation_pixels": 2,
        }
        common = dict(
            model=ZeroModel(),
            loader=[batch],
            device=torch.device("cpu"),
            max_disp=192,
            amp=False,
            logger=logging.getLogger("test"),
            epe_threshold=None,
        )
        standard = validate(evaluation_protocol="standard", **common)
        tradition = validate(evaluation_protocol="tradition", **common)
        self.assertAlmostEqual(standard["epe"], 10.0)
        self.assertAlmostEqual(tradition["epe"], 105.0)
        self.assertEqual(tradition["valid_pixels"], 2)

    def test_manifest_dataset_reads_scale_mask_and_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.arange(64 * 128 * 3, dtype=np.uint8).reshape(64, 128, 3)
            Image.fromarray(image).save(root / "left.png")
            Image.fromarray(np.roll(image, -4, axis=1)).save(root / "right.png")
            np.save(root / "disp.npy", np.full((64, 128), 8.0, dtype=np.float32))
            mask = np.ones((64, 128), dtype=np.uint8) * 255
            mask[:, :8] = 0
            Image.fromarray(mask).save(root / "valid.png")
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["left", "right", "disparity", "valid", "split", "name", "disp_scale"],
                )
                writer.writeheader()
                for split in ("train", "val"):
                    writer.writerow(
                        {
                            "left": "left.png",
                            "right": "right.png",
                            "disparity": "disp.npy",
                            "valid": "valid.png",
                            "split": split,
                            "name": f"sample_{split}",
                            "disp_scale": 2,
                        }
                    )

            dataset = ManifestStereoDataset(
                root / "manifest.csv", "train", crop_size=(32, 64), training=True
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["left"].shape), (3, 32, 64))
            self.assertEqual(tuple(sample["disparity"].shape), (1, 32, 64))
            self.assertAlmostEqual(float(sample["disparity"].median()), 4.0)
            self.assertGreater(int(sample["valid"].sum()), 0)

    def test_synthetic_dataset_is_deterministic(self):
        first = SyntheticShiftDataset(length=1, height=64, width=256, seed=7)[0]
        second = SyntheticShiftDataset(length=1, height=64, width=256, seed=7)[0]
        self.assertTrue(torch.equal(first["left"], second["left"]))
        self.assertTrue(torch.equal(first["right"], second["right"]))
        self.assertTrue(torch.equal(first["disparity"], second["disparity"]))

    def test_training_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=2)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            path = Path(directory) / "checkpoint.pth"
            save_training_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=3,
                global_step=9,
                best_epe=0.75,
                config={"dataset": "test"},
            )
            checkpoint = safe_torch_load(path)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["global_step"], 9)
            self.assertAlmostEqual(checkpoint["best_epe"], 0.75)
            self.assertEqual(set(checkpoint["model"]), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
