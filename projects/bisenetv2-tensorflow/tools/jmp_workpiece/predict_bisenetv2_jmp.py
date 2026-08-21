#!/usr/bin/env python3
"""Run a frozen JMP BiSeNetV2 graph on an image directory."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_pb", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_glob", default="*.png")
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--uncertain_threshold", type=float, default=0.75)
    parser.add_argument(
        "--foreground_threshold",
        type=float,
        default=0.5,
        help="Threshold applied to the foreground probability map.",
    )
    parser.add_argument("--contact_sheet_samples", type=int, default=30)
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Save foreground-class probabilities as float32 NumPy arrays.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def make_overlay(image, mask):
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.45 * overlay[foreground]
        + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def save_contact_sheet(records, output_path, sample_count):
    count = min(sample_count, len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        record = records[int(index)]
        image = cv2.imread(record["prepared_image"], cv2.IMREAD_COLOR)
        overlay = cv2.imread(record["prepared_overlay"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["prepared_mask"], cv2.IMREAD_GRAYSCALE)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        tile = np.hstack([image, overlay, mask_bgr])
        label = (
            f"{record['name']} fg={float(record['foreground_fraction']):.3f} "
            f"unc={float(record['uncertain_fraction']):.3f}"
        )
        cv2.putText(
            tile,
            label,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(tile, (432, 256), interpolation=cv2.INTER_AREA))

    columns = 3
    rows = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start:start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    if not 0.0 < args.foreground_threshold < 1.0:
        raise ValueError("--foreground_threshold must be in (0, 1)")
    if not 0.5 < args.uncertain_threshold < 1.0:
        raise ValueError("uncertain_threshold must be between 0.5 and 1.0")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    model_pb = Path(args.model_pb).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new version to preserve predictions."
        )
    image_paths = sorted(input_dir.glob(args.input_glob))
    if not image_paths:
        raise ValueError(f"No inputs match {args.input_glob!r} in {input_dir}")

    (output_dir / "masks").mkdir(parents=True, exist_ok=False)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=False)
    (output_dir / "prepared_inputs").mkdir(parents=True, exist_ok=False)
    if args.save_probabilities:
        (output_dir / "probabilities").mkdir(parents=True, exist_ok=False)

    try:
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_pb.read_bytes())
            tf.import_graph_def(graph_def, name="")
        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        probability_tensor = graph.get_tensor_by_name("final_probability:0")
        output_tensor = graph.get_tensor_by_name("final_output:0")

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        records = []
        with tf.Session(graph=graph, config=session_config) as session:
            for image_path in image_paths:
                source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if source is None:
                    raise FileNotFoundError(f"Unable to read {image_path}")
                source_height, source_width = source.shape[:2]
                if (source_width, source_height) == (args.width, args.height):
                    prepared = source
                else:
                    prepared = cv2.resize(
                        source,
                        (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                batch = ((rgb - 0.5) / 0.5)[None]
                probability, prediction = session.run(
                    [probability_tensor, output_tensor],
                    feed_dict={input_tensor: batch},
                )
                if probability.shape != (1, args.height, args.width, 2):
                    raise ValueError(f"Unexpected probability shape for {image_path}: {probability.shape}")
                if prediction.shape != (args.height, args.width):
                    raise ValueError(f"Unexpected mask shape for {image_path}: {prediction.shape}")
                if not np.isfinite(probability).all():
                    raise FloatingPointError(f"Non-finite probability for {image_path}")
                prediction = probability[0, :, :, 1] >= args.foreground_threshold
                classes = set(np.unique(prediction.astype(np.uint8)).tolist())
                if not classes <= {0, 1}:
                    raise ValueError(f"Unexpected classes for {image_path}: {classes}")

                mask = prediction.astype(np.uint8) * 255
                overlay = make_overlay(prepared, mask)
                prepared_image = output_dir / "prepared_inputs" / image_path.name
                mask_path = output_dir / "masks" / image_path.name
                overlay_path = output_dir / "overlays" / f"{image_path.stem}.jpg"
                probability_path = output_dir / "probabilities" / f"{image_path.stem}.npy"
                if not cv2.imwrite(str(prepared_image), prepared):
                    raise OSError(f"Failed to write {prepared_image}")
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Failed to write {mask_path}")
                if not cv2.imwrite(
                    str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94]
                ):
                    raise OSError(f"Failed to write {overlay_path}")
                if args.save_probabilities:
                    np.save(
                        str(probability_path),
                        probability[0, :, :, 1].astype(np.float32),
                        allow_pickle=False,
                    )

                max_probability = np.max(probability[0], axis=-1)
                records.append(
                    {
                        "name": image_path.stem,
                        "source_image": str(image_path),
                        "prepared_image": str(prepared_image),
                        "mask": str(mask_path.relative_to(output_dir)),
                        "prepared_mask": str(mask_path),
                        "overlay": str(overlay_path.relative_to(output_dir)),
                        "prepared_overlay": str(overlay_path),
                        "source_width": source_width,
                        "source_height": source_height,
                        "foreground_fraction": float(np.mean(prediction == 1)),
                        "mean_foreground_probability": float(np.mean(probability[0, :, :, 1])),
                        "mean_confidence": float(np.mean(max_probability)),
                        "uncertain_fraction": float(
                            np.mean(max_probability < args.uncertain_threshold)
                        ),
                        "probability_sum_max_error": float(
                            np.max(np.abs(np.sum(probability[0], axis=-1) - 1.0))
                        ),
                        "mask_sha256": sha256_file(mask_path),
                    }
                )
                if args.save_probabilities:
                    records[-1]["probability"] = str(
                        probability_path.relative_to(output_dir)
                    )
                    records[-1]["probability_sha256"] = sha256_file(probability_path)

        fieldnames = [
            "name",
            "source_image",
            "mask",
            "overlay",
            "source_width",
            "source_height",
            "foreground_fraction",
            "mean_foreground_probability",
            "mean_confidence",
            "uncertain_fraction",
            "probability_sum_max_error",
            "mask_sha256",
        ]
        if args.save_probabilities:
            fieldnames.extend(["probability", "probability_sha256"])
        with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in fieldnames})

        save_contact_sheet(
            records,
            output_dir / "predictions_contact_sheet.jpg",
            args.contact_sheet_samples,
        )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_pb": str(model_pb),
            "model_pb_sha256": sha256_file(model_pb),
            "input_dir": str(input_dir),
            "input_glob": args.input_glob,
            "count": len(records),
            "model_size": {"width": args.width, "height": args.height},
            "preprocess": "BGR->RGB, /255, (x-0.5)/0.5",
            "output_classes": {"0": "background", "1": "workpiece"},
            "foreground_threshold": args.foreground_threshold,
            "uncertain_threshold": args.uncertain_threshold,
            "probabilities_saved": args.save_probabilities,
            "probability_format": (
                "float32 NumPy array containing foreground-class probability at model resolution"
                if args.save_probabilities
                else None
            ),
            "foreground_fraction": summarize(
                [record["foreground_fraction"] for record in records]
            ),
            "mean_confidence": summarize(
                [record["mean_confidence"] for record in records]
            ),
            "uncertain_fraction": summarize(
                [record["uncertain_fraction"] for record in records]
            ),
            "probability_sum_max_error": max(
                record["probability_sum_max_error"] for record in records
            ),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
