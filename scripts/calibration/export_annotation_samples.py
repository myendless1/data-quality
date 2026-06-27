#!/usr/bin/env python3
"""Export random first-frame samples for bbox annotation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from calibration_common import (
    DEFAULT_CAMERA,
    DEFAULT_KEYPOSE_CSV,
    DEFAULT_OUTPUT_DIR,
    decode_hdf5_frame,
    keypose_dataframe,
    resized_rgb,
    write_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypose-csv", type=Path, default=DEFAULT_KEYPOSE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    df = keypose_dataframe(args.keypose_csv)
    if args.count > len(df):
        raise ValueError(f"Requested {args.count} samples, but only {len(df)} rows are available")
    indices = rng.sample(list(df.index), args.count)

    samples = []
    for sample_index, row_index in enumerate(indices):
        row = df.loc[row_index]
        task = str(row["task"])
        episode_id = int(row["episode_id"])
        image_name = f"{sample_index:03d}_{task}_episode_{episode_id:04d}_{args.camera}_frame0.jpg"
        image_path = sample_dir / image_name
        if args.overwrite or not image_path.exists():
            image = resized_rgb(decode_hdf5_frame(Path(row["file"]), args.camera, 0))
            image.save(image_path, quality=94)
        samples.append(
            {
                "sample_id": f"{sample_index:03d}",
                "task": task,
                "episode_id": episode_id,
                "file": str(row["file"]),
                "camera": args.camera,
                "frame": 0,
                "image": str(image_path.relative_to(output_dir)),
                "keypoints_world": {
                    "grasp": {
                        "frame": int(row["grasp_frame"]) if row["grasp_frame"] == row["grasp_frame"] else None,
                        "xyz": [float(row["grasp_x"]), float(row["grasp_y"]), float(row["grasp_z"])],
                    },
                    "place": {
                        "frame": int(row["place_frame"]) if row["place_frame"] == row["place_frame"] else None,
                        "xyz": [float(row["place_x"]), float(row["place_y"]), float(row["place_z"])],
                    },
                },
                "annotations": {},
            }
        )

    manifest = {
        "camera": args.camera,
        "width": 1280,
        "height": 720,
        "samples": samples,
    }
    write_json(output_dir / "manifest.json", manifest)
    annotations_path = output_dir / "annotations.json"
    if args.overwrite or not annotations_path.exists():
        write_json(
            annotations_path,
            {
                "camera": args.camera,
                "width": 1280,
                "height": 720,
                "samples": [],
            },
        )
    print(f"Wrote {len(samples)} samples to {sample_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Annotations: {annotations_path}")


if __name__ == "__main__":
    main()
