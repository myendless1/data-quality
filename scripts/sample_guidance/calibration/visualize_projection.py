#!/usr/bin/env python3
"""Project keypose xyz to first frames and draw bbox-center markers."""

from __future__ import annotations

import argparse
import json
import math
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
        "--direction-axis",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="-y",
        help="Tool-frame axis used for direction arrows. Matches sampling_guidance default.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=["column", "row"],
        default="column",
        help="Use rotation matrix columns or rows for direction arrows.",
    )
    parser.add_argument(
        "--direction-world-length",
        type=float,
        default=0.04,
        help="World-space length in meters used before projecting each direction arrow.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=36.0,
        help="Displayed arrow length in image pixels.",
    )
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


def direction_from_rotation(
    rotation: np.ndarray,
    direction_axis: str,
    direction_mode: str,
) -> tuple[float | None, float | None]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[direction_axis[1]]
    sign = 1.0 if direction_axis[0] == "+" else -1.0
    basis = rotation if direction_mode == "column" else rotation.T
    vector = sign * basis[:, axis_idx]
    xy = np.asarray(vector[:2], dtype=float)
    norm = float(np.linalg.norm(xy))
    if norm == 0 or not np.isfinite(norm):
        return None, None
    dx, dy = xy / norm
    return float(dx), float(dy)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 4,
    head: int = 13,
    half: int = 6,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    points = [
        end,
        (end[0] - ux * head + px * half, end[1] - uy * head + py * half),
        (end[0] - ux * head - px * half, end[1] - uy * head - py * half),
    ]
    draw.polygon(points, fill=color)


def draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    delta: tuple[float, float],
    color: tuple[int, int, int],
    arrow_length: float,
) -> None:
    du, dv = delta
    norm = math.hypot(du, dv)
    if norm <= 1e-6 or not np.isfinite(norm):
        return
    scale = arrow_length / norm
    end = (start[0] + du * scale, start[1] + dv * scale)
    draw_arrow(draw, start, end, color)


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
            shifted = apply_model_offset(model, point, rotation)
            uv, z = project_points(
                shifted.reshape(1, 3),
                model.fx,
                model.fy,
                model.cx,
                model.cy,
                np.asarray(model.rvec, dtype=float),
                np.asarray(model.tvec, dtype=float),
            )
            x, y = uv[0]
            if z[0] > 0 and -100 <= x <= model.width + 100 and -100 <= y <= model.height + 100:
                dx, dy = direction_from_rotation(rotation, args.direction_axis, args.direction_mode)
                if dx is not None and dy is not None:
                    end_world = shifted + np.asarray([dx, dy, 0.0], dtype=float) * args.direction_world_length
                    end_uv, end_z = project_points(
                        end_world.reshape(1, 3),
                        model.fx,
                        model.fy,
                        model.cx,
                        model.cy,
                        np.asarray(model.rvec, dtype=float),
                        np.asarray(model.tvec, dtype=float),
                    )
                    if end_z[0] > 0:
                        end_x, end_y = end_uv[0]
                        draw_direction_arrow(
                            draw,
                            (float(x), float(y)),
                            (float(end_x - x), float(end_y - y)),
                            color,
                            args.arrow_length,
                        )
                draw_marker(draw, (float(x), float(y)), color, label)
        name = f"{out_index:03d}_{row['task']}_episode_{int(row['episode_id']):04d}_{args.camera}.jpg"
        image.save(args.output_dir / name, quality=94)
    print(f"Wrote {len(selected)} projection images to {args.output_dir}")


if __name__ == "__main__":
    main()
