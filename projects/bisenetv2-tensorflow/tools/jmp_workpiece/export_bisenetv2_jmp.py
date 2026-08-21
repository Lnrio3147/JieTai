#!/usr/bin/env python3
"""Freeze a trained JMP BiSeNetV2 checkpoint into a TensorFlow PB graph."""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint prefix, for example best.ckpt")
    parser.add_argument("--config", default="./config/jmp_workpiece/jmp_workpiece_bisenetv2.yaml")
    parser.add_argument("--output_pb", required=True)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    return parser.parse_args()


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

    checkpoint = str(Path(args.checkpoint).resolve())
    output_pb = Path(args.output_pb).resolve()
    output_pb.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(config_path=str(Path(args.config).resolve()))

    input_tensor = tf.placeholder(
        tf.float32, [1, args.height, args.width, 3], name="input_tensor"
    )
    model = bisenet_v2.BiseNetV2(phase="test", cfg=cfg)
    prediction = model.inference(input_tensor, name="BiseNetV2", reuse=False)
    probability = tf.get_default_graph().get_tensor_by_name("BiseNetV2/prob:0")
    final_probability = tf.identity(probability, name="final_probability")
    final_output = tf.identity(tf.squeeze(prediction, axis=0), name="final_output")

    saver = tf.train.Saver(tf.global_variables())
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as session:
        saver.restore(session, checkpoint)
        frozen = tf.graph_util.convert_variables_to_constants(
            session,
            tf.get_default_graph().as_graph_def(),
            [final_probability.op.name, final_output.op.name],
        )
        with tf.gfile.GFile(str(output_pb), "wb") as handle:
            handle.write(frozen.SerializeToString())

    metadata = {
        "checkpoint": checkpoint,
        "output_pb": str(output_pb),
        "input": {"name": "input_tensor:0", "shape": [1, args.height, args.width, 3], "dtype": "float32"},
        "outputs": {
            "probability": "final_probability:0",
            "class_mask": "final_output:0",
        },
        "preprocess": "BGR->RGB, /255, (x-0.5)/0.5",
    }
    with output_pb.with_suffix(output_pb.suffix + ".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
