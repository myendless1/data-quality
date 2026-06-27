#!/usr/bin/env python3
"""Sample Astribot episodes with grasp/place farthest-point sampling."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import h5py
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Text file containing candidate HDF5 paths, usually step1 output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/astribot_analysis/fps_sampled_hdf5_files.txt"),
        help="Text file containing one sampled HDF5 path per line.",
    )
    parser.add_argument(
        "--output-path-format",
        choices=["path", "name"],
        default="path",
        help="Write full HDF5 paths or only HDF5 file names to --output.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/astribot_analysis/fps_sampling_manifest.csv"),
        help="CSV manifest containing keypose coordinates and FPS selection order.",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pose-group", default="poses_dict")
    parser.add_argument("--gripper-group", default="poses_dict")
    parser.add_argument(
        "--low-gripper-threshold",
        type=float,
        default=0.11,
        help="Threshold used to find the longest continuous low-gripper interval.",
    )
    parser.add_argument(
        "--z-smooth-window",
        type=int,
        default=15,
        help="Centered moving-average window before finding the last negative z slope.",
    )
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - normalized gripper value before thresholding.",
    )
    parser.add_argument(
        "--fps-dims",
        choices=["xy", "xyz"],
        default="xyz",
        help="Use xy or xyz coordinates for grasp/place distances.",
    )
    parser.add_argument(
        "--initial-sample",
        choices=["center-farthest", "random"],
        default="center-farthest",
        help="How to choose the first selected episode.",
    )
    parser.add_argument(
        "--sampling-scope",
        choices=["per-task", "global"],
        default="per-task",
        help="Sample --num-samples per task, or globally across all tasks.",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    return parser


def read_path_list(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"No candidate paths found in {path}")
    return paths


def episode_id(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def infer_task(path: Path) -> str:
    stem = path.stem
    if "_episode_" in stem:
        return stem.split("_episode_", 1)[0]
    return path.parent.name


def sort_key(path: Path) -> tuple[bool, int | str]:
    episode = episode_id(path)
    return (episode is None, episode if episode is not None else path.stem)


def prepare_gripper_values(raw_gripper: np.ndarray, invert_gripper_value: bool) -> np.ndarray:
    gripper = np.asarray(raw_gripper, dtype=float).reshape(-1)
    finite = gripper[np.isfinite(gripper)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.5:
        gripper = gripper / 100.0
    if invert_gripper_value:
        gripper = 1.0 - gripper
    return gripper


def longest_true_interval(mask: np.ndarray) -> tuple[int, int] | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0 or not np.any(mask):
        return None
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def smooth_centered(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def detect_keypose_frames(
    gripper: np.ndarray,
    z: np.ndarray,
    low_gripper_threshold: float,
    z_smooth_window: int,
) -> dict[str, int | float | None]:
    n_frames = min(len(gripper), len(z))
    gripper = np.asarray(gripper[:n_frames], dtype=float).reshape(-1)
    z = np.asarray(z[:n_frames], dtype=float).reshape(-1)
    interval = longest_true_interval(np.isfinite(gripper) & (gripper < low_gripper_threshold))
    if interval is None:
        return {
            "grasp_frame": None,
            "place_frame": None,
            "low_gripper_start_frame": None,
            "low_gripper_end_frame": None,
            "low_gripper_length": 0,
            "place_smoothed_z_slope": None,
        }

    start, end = interval
    smoothed_z = smooth_centered(z[start : end + 1], z_smooth_window)
    if smoothed_z.size >= 2:
        slopes = np.gradient(smoothed_z)
        negative = np.flatnonzero(np.isfinite(slopes) & (slopes < 0))
    else:
        slopes = np.asarray([], dtype=float)
        negative = np.asarray([], dtype=int)
    place_offset = int(negative[-1]) if negative.size else None
    place_frame = start + place_offset if place_offset is not None else None
    place_slope = float(slopes[place_offset]) if place_offset is not None else None

    return {
        "grasp_frame": int(start),
        "place_frame": int(place_frame) if place_frame is not None else None,
        "low_gripper_start_frame": int(start),
        "low_gripper_end_frame": int(end),
        "low_gripper_length": int(end - start + 1),
        "place_smoothed_z_slope": place_slope,
    }


def pose_xyz(pose: np.ndarray, frame: int | float | None) -> tuple[float, float, float]:
    if frame is None:
        return float("nan"), float("nan"), float("nan")
    frame = int(frame)
    if frame < 0 or frame >= pose.shape[0]:
        return float("nan"), float("nan"), float("nan")
    xyz = np.asarray(pose[frame, :3], dtype=float)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def keypose_record_for_episode(
    path: Path,
    arm: str,
    pose_group: str,
    gripper_group: str,
    low_gripper_threshold: float,
    z_smooth_window: int,
    invert_gripper_value: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "task": infer_task(path),
        "episode_id": episode_id(path),
        "file": str(path),
        "valid_keyposes": False,
        "invalid_reason": "",
    }
    pose_key = f"{pose_group}/astribot_arm_{arm}"
    gripper_key = f"{gripper_group}/astribot_gripper_{arm}"
    try:
        with h5py.File(path, "r") as h5:
            missing = [key for key in (pose_key, gripper_key) if key not in h5]
            if missing:
                row["invalid_reason"] = "missing_dataset:" + ";".join(missing)
                return row
            pose = np.asarray(h5[pose_key], dtype=float)
            gripper = prepare_gripper_values(np.asarray(h5[gripper_key], dtype=float), invert_gripper_value)
    except Exception as exc:
        row["invalid_reason"] = f"read_error:{type(exc).__name__}:{exc}"
        return row

    detection = detect_keypose_frames(
        gripper,
        pose[:, 2],
        low_gripper_threshold=low_gripper_threshold,
        z_smooth_window=z_smooth_window,
    )
    row.update(detection)
    for name in ("grasp", "place"):
        x, y, z = pose_xyz(pose, detection[f"{name}_frame"])
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
        row[f"{name}_z"] = z

    coords = [row[f"{name}_{dim}"] for name in ("grasp", "place") for dim in ("x", "y", "z")]
    row["valid_keyposes"] = all(np.isfinite(float(value)) for value in coords)
    if not row["valid_keyposes"] and not row["invalid_reason"]:
        row["invalid_reason"] = "missing_or_nonfinite_keypose"
    return row


def collect_keypose_records(
    paths: list[Path],
    arm: str,
    pose_group: str,
    gripper_group: str,
    low_gripper_threshold: float,
    z_smooth_window: int,
    invert_gripper_value: bool,
) -> list[dict[str, object]]:
    return [
        keypose_record_for_episode(
            path,
            arm=arm,
            pose_group=pose_group,
            gripper_group=gripper_group,
            low_gripper_threshold=low_gripper_threshold,
            z_smooth_window=z_smooth_window,
            invert_gripper_value=invert_gripper_value,
        )
        for path in sorted(paths, key=sort_key)
    ]


def keypose_matrix(rows: list[dict[str, object]], fps_dims: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    dims = ["x", "y"] if fps_dims == "xy" else ["x", "y", "z"]
    columns = [f"{name}_{dim}" for name in ("grasp", "place") for dim in dims]
    valid_rows = [row for row in rows if row.get("valid_keyposes")]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"Skipping {skipped} candidates with missing/non-finite grasp/place coordinates")
    points = np.asarray([[float(row[column]) for column in columns] for row in valid_rows], dtype=float)
    if points.size == 0:
        raise ValueError("No candidates have finite grasp/place coordinates")
    return points, valid_rows


def points_from_rows(rows: list[dict[str, object]], fps_dims: str) -> np.ndarray:
    dims = ["x", "y"] if fps_dims == "xy" else ["x", "y", "z"]
    columns = [f"{name}_{dim}" for name in ("grasp", "place") for dim in dims]
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def first_index(points: np.ndarray, mode: str, seed: int) -> int:
    if mode == "random":
        return random.Random(seed).randrange(points.shape[0])
    dims = points.shape[1] // 2
    grasp = points[:, :dims]
    place = points[:, dims:]
    grasp_center = np.nanmean(grasp, axis=0, keepdims=True)
    place_center = np.nanmean(place, axis=0, keepdims=True)
    scores = np.linalg.norm(grasp - grasp_center, axis=1) + np.linalg.norm(place - place_center, axis=1)
    return int(np.argmax(scores))


def grasp_place_fps_indices(
    points: np.ndarray,
    num_samples: int,
    initial_sample: str = "center-farthest",
    seed: int = 20260624,
) -> tuple[list[int], list[float]]:
    """FPS where score=max min_dist(grasp)+min_dist(place) to selected episodes."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    sample_count = min(num_samples, points.shape[0])
    dims = points.shape[1] // 2
    grasp = points[:, :dims]
    place = points[:, dims:]

    selected = [first_index(points, initial_sample, seed)]
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    selected_mask[selected[0]] = True
    selection_scores = [float("inf")]

    min_grasp_dist = np.linalg.norm(grasp - grasp[selected[0]], axis=1)
    min_place_dist = np.linalg.norm(place - place[selected[0]], axis=1)

    while len(selected) < sample_count:
        scores = min_grasp_dist + min_place_dist
        scores[selected_mask] = -np.inf
        next_idx = int(np.argmax(scores))
        selected.append(next_idx)
        selection_scores.append(float(scores[next_idx]))
        selected_mask[next_idx] = True
        min_grasp_dist = np.minimum(min_grasp_dist, np.linalg.norm(grasp - grasp[next_idx], axis=1))
        min_place_dist = np.minimum(min_place_dist, np.linalg.norm(place - place[next_idx], axis=1))

    return selected, selection_scores


def format_output_path(path: Path, output_path_format: str) -> str:
    if output_path_format == "name":
        return path.name
    if output_path_format != "path":
        raise ValueError(f"Unsupported output path format: {output_path_format}")
    return str(path)


def select_rows_by_fps(
    valid_rows: list[dict[str, object]],
    num_samples: int,
    fps_dims: str,
    initial_sample: str,
    seed: int,
    sampling_scope: str,
) -> list[dict[str, object]]:
    if sampling_scope == "global":
        groups = [("global", valid_rows)]
    elif sampling_scope == "per-task":
        tasks = sorted({str(row["task"]) for row in valid_rows})
        groups = [(task, [row for row in valid_rows if str(row["task"]) == task]) for task in tasks]
    else:
        raise ValueError(f"Unsupported sampling scope: {sampling_scope}")

    selected_rows: list[dict[str, object]] = []
    global_order = 1
    for group_offset, (scope_name, group_rows) in enumerate(groups):
        if not group_rows:
            continue
        points = points_from_rows(group_rows, fps_dims)
        selected_indices, scores = grasp_place_fps_indices(
            points,
            num_samples=num_samples,
            initial_sample=initial_sample,
            seed=seed + group_offset,
        )
        for task_order, (idx, score) in enumerate(zip(selected_indices, scores), start=1):
            row = group_rows[idx]
            row["selected"] = True
            row["selection_scope"] = scope_name
            row["selection_order"] = global_order
            row["task_selection_order"] = task_order
            row["fps_score_at_selection"] = score
            selected_rows.append(row)
            global_order += 1
    return selected_rows


def filter_by_grasp_place_fps(
    input_path: Path,
    output: Path,
    manifest: Path,
    num_samples: int,
    arm: str = "right",
    pose_group: str = "poses_dict",
    gripper_group: str = "poses_dict",
    low_gripper_threshold: float = 0.11,
    z_smooth_window: int = 15,
    invert_gripper_value: bool = True,
    fps_dims: str = "xyz",
    initial_sample: str = "center-farthest",
    seed: int = 20260624,
    output_path_format: str = "path",
    sampling_scope: str = "per-task",
) -> tuple[list[Path], list[dict[str, object]]]:
    """Step 2: sample candidates by grasp/place FPS and write outputs."""
    candidates = read_path_list(input_path)
    rows = collect_keypose_records(
        candidates,
        arm=arm,
        pose_group=pose_group,
        gripper_group=gripper_group,
        low_gripper_threshold=low_gripper_threshold,
        z_smooth_window=z_smooth_window,
        invert_gripper_value=invert_gripper_value,
    )

    _, valid_rows = keypose_matrix(rows, fps_dims)

    for row in valid_rows:
        row["selected"] = False
        row["selection_scope"] = ""
        row["selection_order"] = ""
        row["task_selection_order"] = ""
        row["fps_score_at_selection"] = ""
    selected_rows = select_rows_by_fps(
        valid_rows,
        num_samples=num_samples,
        fps_dims=fps_dims,
        initial_sample=initial_sample,
        seed=seed,
        sampling_scope=sampling_scope,
    )
    selected_paths = [Path(str(row["file"])) for row in selected_rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(format_output_path(path, output_path_format) for path in selected_paths) + "\n")
    fieldnames = [
        "task",
        "episode_id",
        "valid_keyposes",
        "selected",
        "selection_scope",
        "selection_order",
        "task_selection_order",
        "fps_score_at_selection",
        "invalid_reason",
        "grasp_frame",
        "grasp_x",
        "grasp_y",
        "grasp_z",
        "place_frame",
        "place_x",
        "place_y",
        "place_z",
        "low_gripper_start_frame",
        "low_gripper_end_frame",
        "low_gripper_length",
        "place_smoothed_z_slope",
        "file",
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)

    return selected_paths, valid_rows


def main() -> None:
    args = build_argparser().parse_args()
    selected_paths, manifest_df = filter_by_grasp_place_fps(
        input_path=args.input,
        output=args.output,
        manifest=args.manifest,
        num_samples=args.num_samples,
        arm=args.arm,
        pose_group=args.pose_group,
        gripper_group=args.gripper_group,
        low_gripper_threshold=args.low_gripper_threshold,
        z_smooth_window=args.z_smooth_window,
        invert_gripper_value=args.invert_gripper_value,
        fps_dims=args.fps_dims,
        initial_sample=args.initial_sample,
        seed=args.seed,
        output_path_format=args.output_path_format,
        sampling_scope=args.sampling_scope,
    )
    print(f"Wrote {len(selected_paths)} sampled HDF5 paths to {args.output}")
    print(f"Wrote FPS manifest for {len(manifest_df)} finite candidates to {args.manifest}")


if __name__ == "__main__":
    main()
