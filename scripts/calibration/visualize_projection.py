#!/usr/bin/env python3
"""Project keypose xyz to first frames and draw bbox-center markers."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from calibration_common import (
    DEFAULT_CAMERA,
    DEFAULT_KEYPOSE_CSV,
    DEFAULT_OUTPUT_DIR,
    apply_model_offset,
    decode_hdf5_frame,
    keypose_dataframe,
    load_model,
    pose_rotation_at,
    project_points,
    resized_rgb,
    task_episode_path,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypose-csv", type=Path, default=DEFAULT_KEYPOSE_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "camera_model.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "projection_check")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Optional annotation manifest whose task/episode pairs should be excluded.",
    )
    return parser


def font(size: int):
    for candidate in [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], label: str) -> None:
    x, y = xy
    r = 10
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
    draw.line((x - 18, y, x + 18, y), fill=color, width=3)
    draw.line((x, y - 18, x, y + 18), fill=color, width=3)
    draw.text((x + 14, y - 18), label, fill=color, font=font(22), stroke_width=2, stroke_fill=(0, 0, 0))


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)
    model = load_model(args.model)
    df = keypose_dataframe(args.keypose_csv)
    if args.exclude_manifest:
        manifest = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
        excluded = {(str(item["task"]), int(item["episode_id"])) for item in manifest.get("samples", [])}
        df = df[
            ~df.apply(lambda row: (str(row["task"]), int(row["episode_id"])) in excluded, axis=1)
        ].reset_index(drop=True)
    selected = rng.sample(list(df.index), min(args.count, len(df)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for out_index, row_index in enumerate(selected):
        row = df.loc[row_index]
        image = resized_rgb(decode_hdf5_frame(Path(row["file"]), args.camera, 0), (model.width, model.height))
        draw = ImageDraw.Draw(image)
        for label, color in [
            ("grasp", (53, 167, 255)),
            ("place", (255, 178, 56)),
        ]:
            task = str(row["task"])
            episode_id = int(row["episode_id"])
            frame = int(row[f"{label}_frame"])
            rotation = pose_rotation_at(task_episode_path(task, episode_id), frame)
            point = np.asarray([float(row[f"{label}_x"]), float(row[f"{label}_y"]), float(row[f"{label}_z"])])
            shifted = apply_model_offset(model, point, rotation).reshape(1, 3)
            uv, z = project_points(
                shifted,
                model.fx,
                model.fy,
                model.cx,
                model.cy,
                np.asarray(model.rvec, dtype=float),
                np.asarray(model.tvec, dtype=float),
            )
            x, y = uv[0]
            if z[0] > 0 and -100 <= x <= model.width + 100 and -100 <= y <= model.height + 100:
                draw_marker(draw, (float(x), float(y)), color, label)
        name = f"{out_index:03d}_{row['task']}_episode_{int(row['episode_id']):04d}_{args.camera}.jpg"
        image.save(args.output_dir / name, quality=94)
    print(f"Wrote {len(selected)} projection images to {args.output_dir}")


if __name__ == "__main__":
    main()
