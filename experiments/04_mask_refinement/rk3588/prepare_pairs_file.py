#!/usr/bin/env python3
"""Create a deterministic stereo-pair list for board_benchmark.py."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "training" / "JMP-LF6020-ETH3D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene-prefix", default="fdjyp_3_")
    parser.add_argument("--output", type=Path, default=Path("build/fdjyp3_pairs.txt"))
    parser.add_argument(
        "--relative-to",
        type=Path,
        help="Write paths relative to this directory (normally the board data root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    pairs = []
    for left in sorted(dataset_root.glob(f"{args.scene_prefix}*/im0.png")):
        right = left.with_name("im1.png")
        if right.is_file():
            pairs.append((left.resolve(), right.resolve()))
    if not pairs:
        raise ValueError(
            f"No pairs with prefix {args.scene_prefix!r} below {dataset_root}"
        )

    relative_to = args.relative_to.expanduser().resolve() if args.relative_to else None
    lines = []
    for left, right in pairs:
        if relative_to is not None:
            left = left.relative_to(relative_to)
            right = right.relative_to(relative_to)
        lines.append(f"{shlex.quote(str(left))} {shlex.quote(str(right))}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} pairs to {output}")


if __name__ == "__main__":
    main()
