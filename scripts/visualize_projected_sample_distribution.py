#!/usr/bin/env python3
"""Visualize final sampled grasp/place distributions through the calibrated projection model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL = Path("outputs/calibration_bbox_labeled/camera_model_shared_tool.json")
DEFAULT_BACKGROUND = Path("image.png")
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Step2 FPS manifest containing selected rows and grasp/place xyz.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/astribot_filter/projection_distribution"),
    )
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pose-group", default="poses_dict")
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument(
        "--background",
        type=Path,
        default=DEFAULT_BACKGROUND,
        help="Image used as the plot background. Defaults to project-root image.png.",
    )
    parser.add_argument(
        "--group-by",
        choices=["task", "keypose"],
        default="task",
        help="Create one image per task or one image per keypose.",
    )
    parser.add_argument(
        "--direction-axis",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="-y",
        help="Tool-frame axis used for per-point direction arrows. Matches sampling_guidance default.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=["column", "row"],
        default="column",
        help="Use rotation matrix columns or rows for per-point direction arrows.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=24.0,
        help="Per-point direction arrow length in image pixels.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def rotation_matrix_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def pose_rotation_at(path: Path, frame: int, arm: str, pose_group: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        pose = np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"][int(frame)], dtype=float)
    return rotation_matrix_from_xyzw(pose[3:7])


def pose_at(path: Path, frame: int, arm: str, pose_group: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"][int(frame)], dtype=float)


def shared_tool_offset(model: dict[str, Any]) -> np.ndarray:
    offsets = model.get("object_offsets", model.get("object_offsets_world", {})) or {}
    return np.asarray(offsets.get("shared_tool_frame", [0.0, 0.0, 0.0]), dtype=float)


def project_point(point_world: np.ndarray, model: dict[str, Any]) -> tuple[float, float, float]:
    intr = model["intrinsics"]
    extr = model["extrinsics"]
    rotation = rodrigues(np.asarray(extr["rotation_vector"], dtype=float))
    tvec = np.asarray(extr["translation"], dtype=float).reshape(3)
    point_cam = rotation @ point_world.reshape(3) + tvec
    depth = float(point_cam[2])
    u = float(intr["fx"] * point_cam[0] / depth + intr["cx"])
    v = float(intr["fy"] * point_cam[1] / depth + intr["cy"])
    return u, v, depth


def selected_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [row for row in rows if str(row.get("selected", "")).lower() == "true"]
    if not selected:
        raise ValueError(f"No selected rows found in {manifest}")
    return sorted(selected, key=lambda row: int(float(row.get("selection_order") or 0)))


def direction_from_pose(
    pose: np.ndarray,
    direction_axis: str,
    direction_mode: str,
) -> tuple[float | None, float | None]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[direction_axis[1]]
    sign = 1.0 if direction_axis[0] == "+" else -1.0
    rotation = rotation_matrix_from_xyzw(pose[3:7])
    basis = rotation if direction_mode == "column" else rotation.T
    vector = sign * basis[:, axis_idx]
    xy = np.asarray(vector[:2], dtype=float)
    norm = float(np.linalg.norm(xy))
    if norm == 0 or not np.isfinite(norm):
        return None, None
    dx, dy = xy / norm
    return float(dx), float(dy)


def projected_rows(
    manifest_rows: list[dict[str, str]],
    model: dict[str, Any],
    arm: str,
    pose_group: str,
    direction_axis: str,
    direction_mode: str,
) -> list[dict[str, object]]:
    offset_tool = shared_tool_offset(model)
    width = int(model.get("width", 1280))
    height = int(model.get("height", 720))
    rows: list[dict[str, object]] = []
    for row in manifest_rows:
        path = Path(row["file"])
        for keypose in ("grasp", "place"):
            frame = int(float(row[f"{keypose}_frame"]))
            point = np.asarray(
                [float(row[f"{keypose}_x"]), float(row[f"{keypose}_y"]), float(row[f"{keypose}_z"])],
                dtype=float,
            )
            pose = pose_at(path, frame, arm=arm, pose_group=pose_group)
            rotation_world_tool = rotation_matrix_from_xyzw(pose[3:7])
            shifted = point + rotation_world_tool @ offset_tool
            u, v, depth = project_point(shifted, model)
            direction_u = None
            direction_v = None
            dx, dy = direction_from_pose(pose, direction_axis, direction_mode)
            if dx is not None and dy is not None:
                end_world = shifted + np.asarray([dx, dy, 0.0], dtype=float) * 0.04
                end_u, end_v, end_depth = project_point(end_world, model)
                if end_depth > 0:
                    direction_u = float(end_u - u)
                    direction_v = float(end_v - v)
            rows.append(
                {
                    "keypose": keypose,
                    "task": row["task"],
                    "episode_id": row["episode_id"],
                    "selection_order": row["selection_order"],
                    "u": u,
                    "v": v,
                    "depth": depth,
                    "in_image": depth > 0 and 0 <= u <= width and 0 <= v <= height,
                    "file": row["file"],
                    "direction_u": direction_u,
                    "direction_v": direction_v,
                }
            )
    return rows


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


def draw_distribution(
    rows: list[dict[str, object]],
    group_name: str,
    group_by: str,
    output_path: Path,
    width: int,
    height: int,
    point_radius: int,
    background: Path | None,
    arrow_length: float,
) -> None:
    if background is not None and background.exists():
        canvas = Image.open(background).convert("RGB").resize((width, height), RESAMPLE_LANCZOS)
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        canvas = Image.blend(canvas, overlay, 0.18)
    else:
        canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(24)
    small_font = font(15)
    task_colors = {
        "centrifuge": (53, 167, 255),
        "multidrop": (255, 178, 56),
    }
    keypose_colors = {
        "grasp": (53, 167, 255),
        "place": (255, 178, 56),
    }
    colors = keypose_colors if group_by == "task" else task_colors
    color_field = "keypose" if group_by == "task" else "task"
    fallback = (245, 245, 245)

    subset = [row for row in rows if str(row[group_by]) == group_name]
    in_image = [row for row in subset if row["in_image"]]
    out_image_count = len(subset) - len(in_image)
    for row in in_image:
        u = float(row["u"])
        v = float(row["v"])
        color = colors.get(str(row[color_field]), fallback)
        direction_u = row.get("direction_u")
        direction_v = row.get("direction_v")
        if direction_u is not None and direction_v is not None:
            du = float(direction_u)
            dv = float(direction_v)
            norm = math.hypot(du, dv)
            if norm > 1e-6 and np.isfinite(norm):
                scale = arrow_length / norm
                end = (u + du * scale, v + dv * scale)
                draw_arrow(draw, (u, v), end, color, width=2, head=6, half=3)
        r = point_radius
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=(0, 0, 0), width=2)
        draw.ellipse((u - r + 1, v - r + 1, u + r - 1, v + r - 1), outline=(255, 255, 255), width=1)

    title = f"{group_name} projected distribution  n={len(subset)}"
    if out_image_count:
        title += f"  out_of_image={out_image_count}"
    draw.text((20, 18), title, fill=(255, 255, 255), font=title_font, stroke_width=3, stroke_fill=(0, 0, 0))

    legend_x = width - 230
    legend_y = 18
    for idx, (label, color) in enumerate(colors.items()):
        y = legend_y + idx * 24
        draw.ellipse((legend_x, y + 4, legend_x + 12, y + 16), fill=color, outline=(0, 0, 0))
        draw.text(
            (legend_x + 20, y),
            label,
            fill=(255, 255, 255),
            font=small_font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def visualize_projected_distribution(
    manifest: Path,
    model_path: Path,
    output_dir: Path,
    arm: str = "right",
    pose_group: str = "poses_dict",
    point_radius: int = 5,
    background: Path | None = DEFAULT_BACKGROUND,
    group_by: str = "task",
    direction_axis: str = "-y",
    direction_mode: str = "column",
    arrow_length: float = 24.0,
) -> list[Path]:
    model = load_json(model_path)
    rows = projected_rows(
        selected_manifest_rows(manifest),
        model=model,
        arm=arm,
        pose_group=pose_group,
        direction_axis=direction_axis,
        direction_mode=direction_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "projected_points.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "keypose",
            "task",
            "episode_id",
            "selection_order",
            "u",
            "v",
            "depth",
            "in_image",
            "direction_u",
            "direction_v",
            "file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    width = int(model.get("width", 1280))
    height = int(model.get("height", 720))
    output_paths = [csv_path]
    if group_by == "task":
        group_names = sorted({str(row["task"]) for row in rows})
    elif group_by == "keypose":
        group_names = ["grasp", "place"]
    else:
        raise ValueError(f"Unsupported group_by: {group_by}")
    for group_name in group_names:
        image_path = output_dir / f"{group_name}_projected_distribution.png"
        draw_distribution(
            rows,
            group_name,
            group_by,
            image_path,
            width,
            height,
            point_radius,
            background,
            arrow_length,
        )
        output_paths.append(image_path)
    return output_paths


def main() -> None:
    args = build_argparser().parse_args()
    output_paths = visualize_projected_distribution(
        manifest=args.manifest,
        model_path=args.model,
        output_dir=args.output_dir,
        arm=args.arm,
        pose_group=args.pose_group,
        point_radius=args.point_radius,
        background=args.background,
        group_by=args.group_by,
        direction_axis=args.direction_axis,
        direction_mode=args.direction_mode,
        arrow_length=args.arrow_length,
    )
    print("Wrote projected distribution outputs:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
