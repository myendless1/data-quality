"""Shared helpers for Astribot HDF5 data analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TASK_DIRS = {
    "centrifuge": Path("/media/damoxing/datasets/astribot_tasks/myendless/centrifuge"),
    "multidrop": Path("/media/damoxing/datasets/astribot_tasks/myendless/multidrop"),
}


@dataclass(frozen=True)
class KeyPose:
    frame: int | None
    x: float | None
    y: float | None
    z: float | None
    direction_x: float | None
    direction_y: float | None
    direction_angle_rad: float | None


def parse_task_dirs(values: Iterable[str]) -> dict[str, Path]:
    task_dirs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            path = Path(value)
            task_dirs[path.name] = path
        else:
            name, path = value.split("=", 1)
            task_dirs[name] = Path(path)
    return task_dirs


def list_episode_files(task_dir: Path) -> list[Path]:
    return sorted(
        task_dir.glob("*.hdf5"),
        key=lambda p: int(re.search(r"episode_(\d+)", p.stem).group(1))
        if re.search(r"episode_(\d+)", p.stem)
        else p.stem,
    )


def episode_id(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def build_task_files(task_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    task_files = {task: list_episode_files(path) for task, path in task_dirs.items()}
    for task, files in task_files.items():
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found for {task}: {task_dirs[task]}")
    return task_files


def count_frames(h5: h5py.File) -> int:
    if "time" in h5:
        return int(h5["time"].shape[0])
    return int(h5["poses_dict/merge_pose"].shape[0])


def duration_seconds(h5: h5py.File) -> float | None:
    if "time" not in h5 or h5["time"].shape[0] < 2:
        return None
    time = np.asarray(h5["time"])
    return float(time[-1] - time[0])


def summarize_frames(task_files: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for task, files in task_files.items():
        for path in files:
            with h5py.File(path, "r") as h5:
                frames = count_frames(h5)
                duration = duration_seconds(h5)
            rows.append(
                {
                    "task": task,
                    "episode_id": episode_id(path),
                    "file": str(path),
                    "frames": frames,
                    "duration_s": duration,
                    "fps_est": frames / duration if duration and duration > 0 else None,
                }
            )
    return pd.DataFrame(rows)


def frame_summary_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    return (
        frame_df.groupby("task")["frames"]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .reset_index()
    )


def plot_frame_histograms(frame_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for task, group in frame_df.groupby("task"):
        ax.hist(group["frames"], bins=30, alpha=0.55, label=f"{task} (n={len(group)})")
    ax.set_xlabel("frames")
    ax.set_ylabel("episode count")
    ax.set_title("Episode frame distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def classify_gripper(value: float, closed_threshold: float, open_threshold: float) -> str | None:
    if value < closed_threshold:
        return "closed"
    if value > open_threshold:
        return "open"
    return None


def find_last_transition_end(
    gripper: np.ndarray,
    from_state: str,
    to_state: str,
    closed_threshold: float = 0.4,
    open_threshold: float = 0.95,
) -> int | None:
    """Return the final frame where a from_state->to_state transition completes."""
    if from_state == to_state:
        raise ValueError("from_state and to_state must differ")

    in_transition = False
    events: list[int] = []
    for idx, value in enumerate(gripper):
        state = classify_gripper(float(value), closed_threshold, open_threshold)
        if state == from_state:
            in_transition = True
        elif state == to_state and in_transition:
            events.append(idx)
            in_transition = False
    return events[-1] if events else None


def prepare_gripper_values(raw_gripper: np.ndarray, invert_gripper_value: bool) -> np.ndarray:
    """Convert gripper readings to the 0..1 convention used by thresholds."""
    gripper = np.asarray(raw_gripper, dtype=float).reshape(-1)
    finite = gripper[np.isfinite(gripper)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.5:
        gripper = gripper / 100.0
    if invert_gripper_value:
        gripper = 1.0 - gripper
    return gripper


def rotation_matrix_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.full((3, 3), np.nan)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def direction_from_pose(
    pose: np.ndarray,
    direction_axis: str = "+x",
    direction_mode: str = "column",
) -> tuple[float, float, float]:
    """Project a configured tool axis to the world xy plane.

    direction_mode="column" uses rotation matrix columns, i.e. local tool axes in
    world coordinates. direction_mode="row" uses the inverse/passive convention,
    which is useful if a dataset stores the opposite transform.
    """
    if direction_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(f"Unsupported direction_axis: {direction_axis}")
    if direction_mode not in {"column", "row"}:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")

    axis_idx = {"x": 0, "y": 1, "z": 2}[direction_axis[1]]
    sign = 1.0 if direction_axis[0] == "+" else -1.0
    rotation = rotation_matrix_from_xyzw(pose[3:7])
    basis = rotation if direction_mode == "column" else rotation.T
    vector = sign * basis[:, axis_idx]
    xy = np.asarray(vector[:2], dtype=float)
    norm = float(np.linalg.norm(xy))
    if norm == 0 or not np.isfinite(norm):
        return float("nan"), float("nan"), float("nan")
    dx, dy = xy / norm
    return float(dx), float(dy), float(math.atan2(dy, dx))


def pose_at_frame(
    h5: h5py.File,
    frame: int | None,
    arm: str,
    pose_group: str,
    direction_axis: str,
    direction_mode: str,
) -> KeyPose:
    if frame is None:
        return KeyPose(None, None, None, None, None, None, None)
    pose = np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"][frame])
    dx, dy, angle = direction_from_pose(pose, direction_axis, direction_mode)
    return KeyPose(
        frame=int(frame),
        x=float(pose[0]),
        y=float(pose[1]),
        z=float(pose[2]),
        direction_x=dx,
        direction_y=dy,
        direction_angle_rad=angle,
    )


def longest_true_interval(mask: np.ndarray) -> tuple[int, int] | None:
    """Return inclusive start/end for the longest True run."""
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
    """Centered moving average with edge padding."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_grasp_place_from_low_gripper_interval(
    gripper: np.ndarray,
    z: np.ndarray,
    closed_threshold: float,
    z_smooth_window: int,
) -> dict[str, int | float | None]:
    """Detect grasp/place frames from the longest low-gripper interval."""
    n_frames = min(len(gripper), len(z))
    gripper = np.asarray(gripper[:n_frames], dtype=float).reshape(-1)
    z = np.asarray(z[:n_frames], dtype=float).reshape(-1)
    interval = longest_true_interval(np.isfinite(gripper) & (gripper < closed_threshold))
    if interval is None:
        return {
            "grasp_frame": None,
            "place_frame": None,
            "low_gripper_start_frame": None,
            "low_gripper_end_frame": None,
            "low_gripper_length": 0,
            "low_gripper_mean": None,
            "low_gripper_var": None,
            "place_smoothed_z_slope": None,
        }

    start, end = interval
    interval_gripper = gripper[start : end + 1]
    interval_z = z[start : end + 1]
    smoothed_z = smooth_centered(interval_z, z_smooth_window)
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
        "low_gripper_mean": float(np.nanmean(interval_gripper)),
        "low_gripper_var": float(np.nanvar(interval_gripper)),
        "place_smoothed_z_slope": place_slope,
    }


def keyposes_for_episode(
    path: Path,
    arm: str = "right",
    pose_group: str = "poses_dict",
    gripper_group: str = "poses_dict",
    low_gripper_threshold: float = 0.11,
    z_smooth_window: int = 15,
    invert_gripper_value: bool = True,
    direction_axis: str = "+x",
    direction_mode: str = "column",
) -> dict[str, KeyPose]:
    keyposes, _ = keyposes_and_interval_stats_for_episode(
        path,
        arm=arm,
        pose_group=pose_group,
        gripper_group=gripper_group,
        low_gripper_threshold=low_gripper_threshold,
        z_smooth_window=z_smooth_window,
        invert_gripper_value=invert_gripper_value,
        direction_axis=direction_axis,
        direction_mode=direction_mode,
    )
    return keyposes


def keyposes_and_interval_stats_for_episode(
    path: Path,
    arm: str = "right",
    pose_group: str = "poses_dict",
    gripper_group: str = "poses_dict",
    low_gripper_threshold: float = 0.11,
    z_smooth_window: int = 15,
    invert_gripper_value: bool = True,
    direction_axis: str = "+x",
    direction_mode: str = "column",
) -> tuple[dict[str, KeyPose], dict[str, int | float | None]]:
    with h5py.File(path, "r") as h5:
        raw_gripper = np.asarray(h5[f"{gripper_group}/astribot_gripper_{arm}"])
        gripper = prepare_gripper_values(raw_gripper, invert_gripper_value)
        pose = np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"], dtype=float)
        detection = detect_grasp_place_from_low_gripper_interval(
            gripper,
            pose[:, 2],
            closed_threshold=low_gripper_threshold,
            z_smooth_window=z_smooth_window,
        )
        keyposes = {
            "grasp": pose_at_frame(
                h5, detection["grasp_frame"], arm, pose_group, direction_axis, direction_mode
            ),
            "place": pose_at_frame(
                h5, detection["place_frame"], arm, pose_group, direction_axis, direction_mode
            ),
        }
        return keyposes, detection


def collect_keypose_records(
    task_files: dict[str, list[Path]],
    arm: str,
    pose_group: str,
    gripper_group: str,
    low_gripper_threshold: float,
    z_smooth_window: int,
    invert_gripper_value: bool,
    direction_axis: str,
    direction_mode: str,
) -> pd.DataFrame:
    rows = []
    for task, files in task_files.items():
        for path in files:
            keyposes, interval_stats = keyposes_and_interval_stats_for_episode(
                path,
                arm=arm,
                pose_group=pose_group,
                gripper_group=gripper_group,
                low_gripper_threshold=low_gripper_threshold,
                z_smooth_window=z_smooth_window,
                invert_gripper_value=invert_gripper_value,
                direction_axis=direction_axis,
                direction_mode=direction_mode,
            )
            row = {
                "task": task,
                "episode_id": episode_id(path),
                "file": str(path),
            }
            row.update(interval_stats)
            for name, pose in keyposes.items():
                row.update(
                    {
                        f"{name}_frame": pose.frame,
                        f"{name}_x": pose.x,
                        f"{name}_y": pose.y,
                        f"{name}_z": pose.z,
                        f"{name}_direction_x": pose.direction_x,
                        f"{name}_direction_y": pose.direction_y,
                        f"{name}_direction_angle_rad": pose.direction_angle_rad,
                        f"{name}_direction_angle_deg": math.degrees(
                            pose.direction_angle_rad
                        )
                        if pose.direction_angle_rad is not None
                        else None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_gripper_interval_stats(keypose_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=160)
    colors = {"centrifuge": "#1f77b4", "multidrop": "#d62728"}
    for task, group in keypose_df.dropna(
        subset=["low_gripper_mean", "low_gripper_var"]
    ).groupby("task"):
        color = colors.get(task)
        axes[0].hist(
            group["low_gripper_mean"],
            bins=32,
            alpha=0.55,
            label=f"{task} (n={len(group)})",
            color=color,
        )
        axes[1].hist(
            group["low_gripper_var"],
            bins=32,
            alpha=0.55,
            label=f"{task} (n={len(group)})",
            color=color,
        )
        axes[2].scatter(
            group["low_gripper_mean"],
            group["low_gripper_var"],
            s=8,
            alpha=0.45,
            label=task,
            color=color,
        )

    axes[0].set_title("Mean distribution")
    axes[0].set_xlabel("mean gripper value in longest low interval")
    axes[0].set_ylabel("episode count")
    axes[1].set_title("Variance distribution")
    axes[1].set_xlabel("variance of gripper value in longest low interval")
    axes[1].set_ylabel("episode count")
    axes[2].set_title("Mean vs variance")
    axes[2].set_xlabel("mean")
    axes[2].set_ylabel("variance")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_xy_limits(
    points: np.ndarray, pad_ratio: float = 0.10, min_span: float = 1e-6
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute padded xy limits. Replace this function to use fixed workspace bounds."""
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    spans = np.maximum(maxs - mins, min_span)
    centers = (mins + maxs) / 2.0
    mins = centers - spans / 2.0
    maxs = centers + spans / 2.0
    pads = spans * pad_ratio
    return (float(mins[0] - pads[0]), float(maxs[0] + pads[0])), (
        float(mins[1] - pads[1]),
        float(maxs[1] + pads[1]),
    )


def keypose_plot_points_from_df(keypose_df: pd.DataFrame) -> np.ndarray:
    """Return points in display coordinates: horizontal=y, vertical=x."""
    arrays = []
    for x_col, y_col in [("grasp_x", "grasp_y"), ("place_x", "place_y")]:
        arrays.append(keypose_df[[y_col, x_col]].to_numpy(dtype=float))
    return np.vstack(arrays)


def draw_pose_rays(
    ax: plt.Axes,
    points: pd.DataFrame,
    horizontal_col: str,
    vertical_col: str,
    direction_horizontal_col: str,
    direction_vertical_col: str,
    label: str,
    color: str,
    ray_length: float,
    alpha: float = 0.7,
) -> None:
    valid = points[
        [horizontal_col, vertical_col, direction_horizontal_col, direction_vertical_col]
    ].dropna()
    if valid.empty:
        return
    horizontal = valid[horizontal_col].to_numpy(dtype=float)
    vertical = valid[vertical_col].to_numpy(dtype=float)
    direction_horizontal = (
        valid[direction_horizontal_col].to_numpy(dtype=float) * ray_length
    )
    direction_vertical = valid[direction_vertical_col].to_numpy(dtype=float) * ray_length
    ax.scatter(
        horizontal,
        vertical,
        s=3,
        c=color,
        label=f"{label} ({len(valid)})",
        alpha=alpha,
    )
    ax.quiver(
        horizontal,
        vertical,
        direction_horizontal,
        direction_vertical,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        color=color,
        alpha=min(alpha + 0.1, 1.0),
    )


def plot_keypose_distribution_on_axis(
    ax: plt.Axes,
    keypose_df: pd.DataFrame,
    title: str,
    xy_limits: tuple[tuple[float, float], tuple[float, float]],
    ray_ratio: float,
) -> None:
    (hmin, hmax), (vmin, vmax) = xy_limits
    ray_length = math.hypot(hmax - hmin, vmax - vmin) * ray_ratio
    draw_pose_rays(
        ax,
        keypose_df,
        "grasp_y",
        "grasp_x",
        "grasp_direction_y",
        "grasp_direction_x",
        "grasp",
        "#1f77b4",
        ray_length,
        alpha=0.5,
    )
    draw_pose_rays(
        ax,
        keypose_df,
        "place_y",
        "place_x",
        "place_direction_y",
        "place_direction_x",
        "place",
        "#d62728",
        ray_length,
        alpha=0.95,
    )
    ax.set_xlim(hmax, hmin)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(title)
    ax.legend(loc="best")


def plot_keypose_distribution(
    keypose_df: pd.DataFrame,
    output_path: Path,
    title: str,
    xy_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ray_ratio: float = 0.01,
) -> None:
    if xy_limits is None:
        xy_limits = compute_xy_limits(keypose_plot_points_from_df(keypose_df))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    plot_keypose_distribution_on_axis(ax, keypose_df, title, xy_limits, ray_ratio)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def decode_image_frame(h5: h5py.File, camera: str, frame: int) -> np.ndarray:
    data = h5[f"images_dict/{camera}/rgb"]
    sizes = np.asarray(h5[f"images_dict/{camera}/rgb_size"]).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))
    start = int(starts[frame])
    end = start + int(sizes[frame])
    image = cv2.imdecode(np.asarray(data[start:end]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode {camera} frame {frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_single_episode_debug(
    episode_path: Path,
    output_path: Path,
    task: str,
    arm: str,
    camera: str,
    pose_group: str,
    gripper_group: str,
    low_gripper_threshold: float,
    z_smooth_window: int,
    invert_gripper_value: bool,
    direction_axis: str,
    direction_mode: str,
    ray_ratio: float = 0.05,
) -> None:
    keyposes = keyposes_for_episode(
        episode_path,
        arm=arm,
        pose_group=pose_group,
        gripper_group=gripper_group,
        low_gripper_threshold=low_gripper_threshold,
        z_smooth_window=z_smooth_window,
        invert_gripper_value=invert_gripper_value,
        direction_axis=direction_axis,
        direction_mode=direction_mode,
    )
    keypose_df = pd.DataFrame(
        [
            {
                "task": task,
                "episode_id": episode_id(episode_path),
                "file": str(episode_path),
                "grasp_frame": keyposes["grasp"].frame,
                "grasp_x": keyposes["grasp"].x,
                "grasp_y": keyposes["grasp"].y,
                "grasp_direction_x": keyposes["grasp"].direction_x,
                "grasp_direction_y": keyposes["grasp"].direction_y,
                "place_frame": keyposes["place"].frame,
                "place_x": keyposes["place"].x,
                "place_y": keyposes["place"].y,
                "place_direction_x": keyposes["place"].direction_x,
                "place_direction_y": keyposes["place"].direction_y,
            }
        ]
    )
    xy_limits = compute_xy_limits(
        keypose_plot_points_from_df(keypose_df), pad_ratio=0.25, min_span=0.10
    )

    with h5py.File(episode_path, "r") as h5:
        fig = plt.figure(figsize=(14, 6), dpi=160)
        axes = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
        for ax, name in zip(axes[:2], ["grasp", "place"]):
            frame = keyposes[name].frame
            if frame is None:
                ax.text(0.5, 0.5, f"{name}: no transition", ha="center", va="center")
                ax.axis("off")
                continue
            image = decode_image_frame(h5, camera, frame)
            ax.imshow(image)
            ax.set_title(f"{name} frame={frame}")
            ax.axis("off")

        plot_keypose_distribution_on_axis(
            axes[2],
            keypose_df,
            f"{task} episode {episode_id(episode_path)}",
            xy_limits,
            ray_ratio=ray_ratio,
        )
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def plot_multi_episode_debug(
    episode_paths: list[Path],
    output_path: Path,
    task: str,
    arm: str,
    camera: str,
    pose_group: str,
    gripper_group: str,
    low_gripper_threshold: float,
    z_smooth_window: int,
    invert_gripper_value: bool,
    direction_axis: str,
    direction_mode: str,
    ray_ratio: float = 0.05,
) -> None:
    rows = []
    keyposes_by_path = {}
    for episode_path in episode_paths:
        keyposes = keyposes_for_episode(
            episode_path,
            arm=arm,
            pose_group=pose_group,
            gripper_group=gripper_group,
            low_gripper_threshold=low_gripper_threshold,
            z_smooth_window=z_smooth_window,
            invert_gripper_value=invert_gripper_value,
            direction_axis=direction_axis,
            direction_mode=direction_mode,
        )
        keyposes_by_path[episode_path] = keyposes
        rows.append(
            {
                "task": task,
                "episode_id": episode_id(episode_path),
                "file": str(episode_path),
                "grasp_frame": keyposes["grasp"].frame,
                "grasp_x": keyposes["grasp"].x,
                "grasp_y": keyposes["grasp"].y,
                "grasp_direction_x": keyposes["grasp"].direction_x,
                "grasp_direction_y": keyposes["grasp"].direction_y,
                "place_frame": keyposes["place"].frame,
                "place_x": keyposes["place"].x,
                "place_y": keyposes["place"].y,
                "place_direction_x": keyposes["place"].direction_x,
                "place_direction_y": keyposes["place"].direction_y,
            }
        )

    all_keypose_df = pd.DataFrame(rows)
    xy_limits = compute_xy_limits(
        keypose_plot_points_from_df(all_keypose_df), pad_ratio=0.25, min_span=0.10
    )

    fig = plt.figure(figsize=(14, 4.2 * len(episode_paths)), dpi=160)
    for row_idx, episode_path in enumerate(episode_paths):
        keyposes = keyposes_by_path[episode_path]
        row_df = all_keypose_df.iloc[[row_idx]]
        axes = [
            fig.add_subplot(len(episode_paths), 3, row_idx * 3 + 1),
            fig.add_subplot(len(episode_paths), 3, row_idx * 3 + 2),
            fig.add_subplot(len(episode_paths), 3, row_idx * 3 + 3),
        ]
        with h5py.File(episode_path, "r") as h5:
            for ax, name in zip(axes[:2], ["grasp", "place"]):
                frame = keyposes[name].frame
                if frame is None:
                    ax.text(0.5, 0.5, f"{name}: no transition", ha="center", va="center")
                    ax.axis("off")
                    continue
                image = decode_image_frame(h5, camera, frame)
                ax.imshow(image)
                ax.set_title(f"ep={episode_id(episode_path)} {name} frame={frame}")
                ax.axis("off")

        plot_keypose_distribution_on_axis(
            axes[2],
            row_df,
            f"{task} episode {episode_id(episode_path)}",
            xy_limits,
            ray_ratio=ray_ratio,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
