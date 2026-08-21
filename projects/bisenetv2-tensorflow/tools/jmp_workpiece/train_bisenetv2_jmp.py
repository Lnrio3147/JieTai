#!/usr/bin/env python3
"""Train the repository BiSeNetV2 on prepared JMP binary masks."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--config", default="./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--end_learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--preview_samples", type=int, default=12)
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional checkpoint prefix used to initialize model weights only.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable TensorFlow's loss-scaled mixed-precision graph rewrite on CUDA.",
    )
    parser.add_argument(
        "--category_balance_power",
        type=float,
        default=0.0,
        help=(
            "Per-epoch category oversampling strength in [0,1]: 0 disables it, "
            "0.5 uses square-root balancing, and 1 fully balances categories."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        choices=[
            "foreground_iou",
            "macro_category_iou",
            "foreground_f2",
            "macro_category_f2",
        ],
        default="foreground_iou",
        help="Validation metric used to save best.ckpt.",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "light", "strong"],
        default="light",
        help="Photometric augmentation preset; validation is always unaugmented.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(dataset_dir, split, limit=None):
    index_path = dataset_dir / "index" / f"{split}.csv"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No {split} records found in {index_path}")
    return records


def load_batch(records, dataset_dir, training, rng, augmentation="light"):
    images = []
    labels = []
    for record in records:
        image = cv2.imread(str(dataset_dir / record["image"]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(dataset_dir / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise FileNotFoundError(f"Missing prepared sample {record['name']}")

        if training and augmentation != "none" and rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if training and augmentation == "light":
            contrast = rng.uniform(0.9, 1.1)
            brightness = rng.uniform(-0.04, 0.04)
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        elif training and augmentation == "strong":
            # The additional domains differ mostly in lighting, exposure,
            # metal colour and camera response.  Keep geometry unchanged and
            # broaden only these photometric factors.
            contrast = rng.uniform(0.78, 1.22)
            brightness = rng.uniform(-0.09, 0.09)
            gamma = rng.uniform(0.78, 1.28)
            channel_gain = np.asarray(
                [rng.uniform(0.88, 1.12) for _ in range(3)], dtype=np.float32
            )
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            image = np.clip(
                np.power(image, gamma) * channel_gain[None, None, :], 0.0, 1.0
            )
            if rng.random() < 0.20:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            if rng.random() < 0.20:
                noise = np.random.normal(0.0, 0.012, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0)
        image = (image - 0.5) / 0.5
        images.append(image)
        labels.append((label > 0).astype(np.int32))
    return np.stack(images), np.stack(labels)


def update_confusion(confusion, labels, predictions):
    encoded = labels.reshape(-1) * 2 + predictions.reshape(-1)
    confusion += np.bincount(encoded, minlength=4).reshape(2, 2)


def compute_metrics(confusion):
    confusion = confusion.astype(np.float64)
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    tp = confusion[1, 1]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    precision = float(tp / max(tp + fp, 1.0))
    recall = float(tp / max(tp + fn, 1.0))
    return {
        "background_iou": float(iou[0]),
        "foreground_iou": float(iou[1]),
        "mean_iou": float(np.mean(iou)),
        "foreground_dice": float((2 * tp) / max(2 * tp + fp + fn, 1.0)),
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f2": float(
            5.0 * precision * recall / max(4.0 * precision + recall, 1e-12)
        ),
        "pixel_accuracy": float(intersection.sum() / max(confusion.sum(), 1.0)),
        "confusion": confusion.astype(np.int64).tolist(),
    }


def iter_batches(records, batch_size, drop_last=False):
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        if drop_last and len(batch) < batch_size:
            break
        yield batch


def balanced_epoch_size(records, power):
    if power <= 0.0:
        return len(records)
    counts = Counter(record.get("category", "") for record in records)
    if "" in counts:
        raise ValueError("Category balancing requires a category column in train.csv")
    maximum = max(counts.values())
    return sum(
        int(math.ceil(count * (maximum / count) ** power))
        for count in counts.values()
    )


def balance_epoch_records(records, power, rng):
    if power <= 0.0:
        return list(records)
    grouped = {}
    for record in records:
        category = record.get("category")
        if not category:
            raise ValueError("Category balancing requires a category column in train.csv")
        grouped.setdefault(category, []).append(record)
    maximum = max(len(group) for group in grouped.values())
    balanced = []
    for category in sorted(grouped):
        group = grouped[category]
        target = int(math.ceil(len(group) * (maximum / len(group)) ** power))
        balanced.extend(group)
        balanced.extend(rng.choice(group) for _ in range(target - len(group)))
    return balanced


def save_prediction_previews(session, tensors, records, dataset_dir, output_dir, sample_count):
    count = min(sample_count, len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int32)
    tiles = []
    preview_dir = output_dir / "predictions"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for index in indices:
        record = records[int(index)]
        batch_images, batch_labels = load_batch([record], dataset_dir, False, rng)
        prediction = session.run(
            tensors["prediction"],
            feed_dict={
                tensors["images"]: batch_images,
                tensors["labels"]: batch_labels,
                tensors["phase"]: "test",
            },
        )[0].astype(np.uint8)
        source_rgb = ((batch_images[0] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
        gt = batch_labels[0].astype(np.uint8) * 255
        pred = prediction * 255
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        tile = np.hstack([source_bgr, gt_bgr, pred_bgr])
        cv2.putText(tile, record["name"], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.imwrite(str(preview_dir / f"{record['name']}.png"), tile)
        tiles.append(cv2.resize(tile, (432, 256), interpolation=cv2.INTER_AREA))

    columns = 3
    rows = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start:start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    cv2.imwrite(str(output_dir / "val_predictions_contact_sheet.jpg"), np.vstack(rows))


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-bisenet-jmp")

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from bisenet_model import bisenet_v2
    from local_utils.config_utils.parse_config_utils import Config

    dataset_dir = Path(args.dataset_dir).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output must be new or empty: {output_dir}")
    if args.batch_size < 2:
        raise ValueError(
            "batch_size must be at least 2 because the BiSeNetV2 context embedding branch "
            "applies BatchNorm to a 1x1 feature map during training."
        )
    if not 0.0 <= args.category_balance_power <= 1.0:
        raise ValueError("category_balance_power must be in [0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
    )
    log = logging.getLogger("bisenetv2_jmp")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    train_records = read_records(dataset_dir, "train", args.max_train_samples)
    val_records = read_records(dataset_dir, "val", args.max_val_samples)
    sample_image = cv2.imread(str(dataset_dir / train_records[0]["image"]), cv2.IMREAD_COLOR)
    height, width = sample_image.shape[:2]
    cfg = Config(config_path=str(config_path))

    images = tf.placeholder(tf.float32, [None, height, width, 3], name="input_tensor")
    labels = tf.placeholder(tf.int32, [None, height, width], name="input_label")
    phase = tf.placeholder(tf.string, shape=(), name="phase")
    model = bisenet_v2.BiseNetV2(phase=phase, cfg=cfg)
    losses = model.compute_loss(images, labels, name="BiseNetV2", reuse=False)
    prediction = model.inference(images, name="BiseNetV2", reuse=True)
    # Capture only model variables before creating global_step and Adam slots.
    # This permits fine-tuning from an older run without inheriting its optimizer
    # state or exhausted learning-rate schedule.
    model_variables = list(tf.global_variables())

    global_step = tf.train.get_or_create_global_step()
    train_samples_per_epoch = balanced_epoch_size(
        train_records, args.category_balance_power
    )
    steps_per_epoch = train_samples_per_epoch // args.batch_size
    total_steps = max(args.epochs * steps_per_epoch, 1)
    learning_rate = tf.train.polynomial_decay(
        args.learning_rate,
        global_step,
        total_steps,
        end_learning_rate=args.end_learning_rate,
        power=0.9,
        name="learning_rate",
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if args.mixed_precision:
        if args.device != "cuda":
            raise ValueError("--mixed_precision is only supported with --device cuda")
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(losses["total_loss"], global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    init_saver = tf.train.Saver(model_variables) if args.init_checkpoint else None
    tensors = {"images": images, "labels": labels, "phase": phase, "prediction": prediction}
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    init_checkpoint = (
        str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
    )
    if init_checkpoint and not Path(init_checkpoint + ".index").is_file():
        raise FileNotFoundError("Initial checkpoint not found: {}".format(init_checkpoint))
    init_checkpoint_files = (
        sorted(Path(init_checkpoint).parent.glob(Path(init_checkpoint).name + ".*"))
        if init_checkpoint
        else []
    )
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "dataset_dir": str(dataset_dir),
        "dataset_metadata_sha256": sha256_file(dataset_dir / "metadata.json"),
        "model_config": str(config_path),
        "model_config_sha256": sha256_file(config_path),
        "image_size": {"width": width, "height": height},
        "train_samples": len(train_records),
        "train_samples_per_epoch": train_samples_per_epoch,
        "val_samples": len(val_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "drop_last_train_batch": True,
        "learning_rate": args.learning_rate,
        "end_learning_rate": args.end_learning_rate,
        "seed": args.seed,
        "device_request": args.device,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "mixed_precision": args.mixed_precision,
        "category_balance_power": args.category_balance_power,
        "selection_metric": args.selection_metric,
        "augmentation": args.augmentation,
        "init_checkpoint": init_checkpoint,
        "init_checkpoint_files": {
            path.name: sha256_file(path) for path in init_checkpoint_files
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    best_selection_value = -1.0
    rng = random.Random(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    with tf.Session(config=session_config) as session, metrics_path.open("w", encoding="utf-8") as metrics_file:
        session.run(tf.global_variables_initializer())
        if init_saver is not None:
            init_saver.restore(session, init_checkpoint)
            log.info(
                "Initialized %d model variables from %s; optimizer state was reset",
                len(model_variables),
                init_checkpoint,
            )
        log.info("TensorFlow %s devices: %s", tf.__version__, session.list_devices())
        log.info("Training samples=%d validation samples=%d size=%dx%d", len(train_records), len(val_records), width, height)

        for epoch in range(1, args.epochs + 1):
            epoch_records = balance_epoch_records(
                train_records, args.category_balance_power, rng
            )
            rng.shuffle(epoch_records)
            train_losses = []
            for batch_records in iter_batches(epoch_records, args.batch_size, drop_last=True):
                batch_images, batch_labels = load_batch(
                    batch_records,
                    dataset_dir,
                    True,
                    rng,
                    augmentation=args.augmentation,
                )
                _, loss_value = session.run(
                    [train_op, losses["total_loss"]],
                    feed_dict={images: batch_images, labels: batch_labels, phase: "train"},
                )
                if not np.isfinite(loss_value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                train_losses.append(float(loss_value))

            val_losses = []
            confusion = np.zeros((2, 2), dtype=np.int64)
            category_confusions = {}
            for batch_records in iter_batches(val_records, args.batch_size):
                batch_images, batch_labels = load_batch(batch_records, dataset_dir, False, rng)
                loss_value, batch_prediction = session.run(
                    [losses["total_loss"], prediction],
                    feed_dict={images: batch_images, labels: batch_labels, phase: "test"},
                )
                val_losses.append(float(loss_value))
                batch_prediction = batch_prediction.astype(np.int32)
                update_confusion(confusion, batch_labels, batch_prediction)
                for record, label, predicted in zip(
                    batch_records, batch_labels, batch_prediction
                ):
                    category = record.get("category")
                    if category:
                        category_confusion = category_confusions.setdefault(
                            category, np.zeros((2, 2), dtype=np.int64)
                        )
                        update_confusion(
                            category_confusion, label[None], predicted[None]
                        )

            metrics = compute_metrics(confusion)
            category_metrics = {
                category: compute_metrics(category_confusion)
                for category, category_confusion in sorted(category_confusions.items())
            }
            metrics["per_category"] = category_metrics
            metrics["macro_category_iou"] = (
                float(
                    np.mean(
                        [
                            category_metric["foreground_iou"]
                            for category_metric in category_metrics.values()
                        ]
                    )
                )
                if category_metrics
                else metrics["foreground_iou"]
            )
            metrics["macro_category_f2"] = (
                float(
                    np.mean(
                        [
                            category_metric["foreground_f2"]
                            for category_metric in category_metrics.values()
                        ]
                    )
                )
                if category_metrics
                else metrics["foreground_f2"]
            )
            metrics.update({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
                "learning_rate": float(session.run(learning_rate)),
            })
            metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metrics_file.flush()
            saver.save(session, str(output_dir / "latest.ckpt"))
            selection_value = metrics[args.selection_metric]
            if selection_value > best_selection_value:
                best_selection_value = selection_value
                saver.save(session, str(output_dir / "best.ckpt"))
                with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, ensure_ascii=False, indent=2)
            log.info(
                "epoch=%d/%d train_loss=%.5f val_loss=%.5f fg_iou=%.5f recall=%.5f macro_cat_iou=%.5f macro_cat_f2=%.5f miou=%.5f dice=%.5f",
                epoch,
                args.epochs,
                metrics["train_loss"],
                metrics["val_loss"],
                metrics["foreground_iou"],
                metrics["foreground_recall"],
                metrics["macro_category_iou"],
                metrics["macro_category_f2"],
                metrics["mean_iou"],
                metrics["foreground_dice"],
            )

        saver.restore(session, str(output_dir / "best.ckpt"))
        save_prediction_previews(
            session, tensors, val_records, dataset_dir, output_dir, args.preview_samples
        )
    log.info(
        "Training complete. Best %s=%.5f",
        args.selection_metric,
        best_selection_value,
    )


if __name__ == "__main__":
    main()
