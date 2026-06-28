#!/usr/bin/env python3
"""Fit camera intrinsics/extrinsics and fixed object offsets from bbox centers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from calibration_common import (
    DEFAULT_OUTPUT_DIR,
    ProjectionModel,
    bbox_center,
    project_points,
    pose_rotation_at,
    read_json,
    task_episode_path,
    write_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "camera_model.json")
    parser.add_argument("--fit-principal-point", action="store_true")
    parser.add_argument("--fix-offsets", action="store_true")
    parser.add_argument(
        "--offset-mode",
        choices=["shared_tool", "label", "task_label"],
        default="shared_tool",
        help=(
            "shared_tool fits one tool-frame offset shared by every task/keypose. "
            "label and task_label keep the older world-frame offset models."
        ),
    )
    parser.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    return parser


def load_observations(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    manifest = read_json(data_dir / "manifest.json", {})
    annotations = read_json(data_dir / "annotations.json", {})
    sample_by_id = {sample["sample_id"]: sample for sample in manifest.get("samples", [])}
    points = []
    pixels = []
    labels = []
    tasks = []
    rotations = []
    for ann in annotations.get("samples", []):
        sample = sample_by_id.get(ann.get("sample_id"))
        if not sample:
            continue
        for label in ("grasp", "place"):
            center = bbox_center(ann.get("bboxes", {}).get(label))
            xyz = sample["keypoints_world"][label]["xyz"]
            frame = sample["keypoints_world"][label].get("frame")
            if center is None or xyz is None:
                continue
            task = str(sample["task"])
            episode_id = int(sample["episode_id"])
            points.append([float(v) for v in xyz])
            pixels.append([float(center[0]), float(center[1])])
            labels.append(0 if label == "grasp" else 1)
            tasks.append(task)
            rotations.append(pose_rotation_at(task_episode_path(task, episode_id), int(frame)))
    if len(points) < 8:
        raise ValueError(f"Need at least 8 annotated boxes; got {len(points)}")
    return (
        np.asarray(points, dtype=float),
        np.asarray(pixels, dtype=float),
        np.asarray(labels, dtype=int),
        tasks,
        np.asarray(rotations, dtype=float),
    )


def offset_keys(tasks: list[str], offset_mode: str) -> list[tuple[str | None, str]]:
    if offset_mode == "shared_tool":
        return [(None, "shared_tool_frame")]
    if offset_mode == "label":
        return [(None, "grasp"), (None, "place")]
    return [(task, label) for task in sorted(set(tasks)) for label in ("grasp", "place")]


def observation_key(task: str, label_id: int, offset_mode: str) -> tuple[str | None, str]:
    if offset_mode == "shared_tool":
        return (None, "shared_tool_frame")
    return (task if offset_mode == "task_label" else None, "grasp" if label_id == 0 else "place")


def pack_initial(
    width: int,
    height: int,
    fit_principal_point: bool,
    fix_offsets: bool,
    keys: list[tuple[str | None, str]],
) -> np.ndarray:
    values = [
        np.log(950.0),
        np.log(950.0),
    ]
    if fit_principal_point:
        values.extend([width / 2.0, height / 2.0])
    values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.4])
    if not fix_offsets:
        values.extend([0.0, 0.0, 0.0] * len(keys))
    return np.asarray(values, dtype=float)


def unpack_params(
    params: np.ndarray,
    width: int,
    height: int,
    fit_principal_point: bool,
    fix_offsets: bool,
    keys: list[tuple[str | None, str]],
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, dict[tuple[str | None, str], np.ndarray]]:
    pos = 0
    fx = float(np.exp(params[pos]))
    fy = float(np.exp(params[pos + 1]))
    pos += 2
    if fit_principal_point:
        cx, cy = float(params[pos]), float(params[pos + 1])
        pos += 2
    else:
        cx, cy = width / 2.0, height / 2.0
    rvec = np.asarray(params[pos : pos + 3], dtype=float)
    tvec = np.asarray(params[pos + 3 : pos + 6], dtype=float)
    pos += 6
    offsets = {}
    for key in keys:
        if fix_offsets:
            offsets[key] = np.zeros(3, dtype=float)
        else:
            offsets[key] = np.asarray(params[pos : pos + 3], dtype=float)
            pos += 3
    return fx, fy, cx, cy, rvec, tvec, offsets


def offsets_to_json(offsets: dict[tuple[str | None, str], np.ndarray], offset_mode: str) -> dict[str, object]:
    if offset_mode == "shared_tool":
        value = offsets[(None, "shared_tool_frame")]
        return {"shared_tool_frame": [float(v) for v in value]}
    if offset_mode == "label":
        return {label: [float(v) for v in value] for (_, label), value in offsets.items()}
    data: dict[str, object] = {}
    for (task, label), value in offsets.items():
        if task is None:
            continue
        if task not in data:
            data[task] = {}
        data[task][label] = [float(v) for v in value]  # type: ignore[index]
    return data


def main() -> None:
    args = build_argparser().parse_args()
    width, height = 1280, 720
    points, pixels, labels, tasks, rotations = load_observations(args.data_dir)
    keys = offset_keys(tasks, args.offset_mode)

    def shifted_points(offsets: dict[tuple[str | None, str], np.ndarray]) -> np.ndarray:
        shifted = points.copy()
        for idx, (task, label_id) in enumerate(zip(tasks, labels)):
            key = observation_key(task, int(label_id), args.offset_mode)
            if args.offset_mode == "shared_tool":
                shifted[idx] += rotations[idx] @ offsets[key]
            else:
                shifted[idx] += offsets[key]
        return shifted

    def residuals(params: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy, rvec, tvec, offsets = unpack_params(
            params, width, height, args.fit_principal_point, args.fix_offsets, keys
        )
        shifted = shifted_points(offsets)
        uv, z = project_points(shifted, fx, fy, cx, cy, rvec, tvec)
        residual = (uv - pixels).reshape(-1)
        bad_depth = np.minimum(z - 0.05, 0.0) * 1000.0
        regularize_offsets = []
        if not args.fix_offsets:
            for key in keys:
                regularize_offsets.extend(offsets[key] / 0.25)
        return np.concatenate([residual, bad_depth, np.asarray(regularize_offsets, dtype=float)])

    x0 = pack_initial(width, height, args.fit_principal_point, args.fix_offsets, keys)
    result = least_squares(
        residuals,
        x0,
        loss=args.loss,
        f_scale=20.0,
        max_nfev=30000,
    )
    fx, fy, cx, cy, rvec, tvec, offsets = unpack_params(
        result.x, width, height, args.fit_principal_point, args.fix_offsets, keys
    )
    shifted = shifted_points(offsets)
    uv, _ = project_points(shifted, fx, fy, cx, cy, rvec, tvec)
    errors = np.linalg.norm(uv - pixels, axis=1)
    model = ProjectionModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rvec=[float(v) for v in rvec],
        tvec=[float(v) for v in tvec],
        offsets=offsets_to_json(offsets, args.offset_mode),
        residual_rmse_px=float(np.sqrt(np.mean(errors**2))),
        residual_median_px=float(np.median(errors)),
        sample_count=int(len(errors)),
    )
    write_json(args.output, model.to_dict())
    print(f"Wrote model: {args.output}")
    print(f"observations={len(errors)} rmse_px={model.residual_rmse_px:.2f} median_px={model.residual_median_px:.2f}")
    print(f"fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"rvec={model.rvec}")
    print(f"tvec={model.tvec}")
    print(f"offsets={model.offsets}")


if __name__ == "__main__":
    main()
