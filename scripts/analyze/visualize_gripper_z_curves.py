#!/usr/bin/env python3
"""Visualize sampled gripper and z trajectories for Astribot HDF5 episodes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from astribot_common import (
    DEFAULT_TASK_DIRS,
    build_task_files,
    episode_id,
    parse_task_dirs,
    prepare_gripper_values,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        action="append",
        default=[],
        help="Task directory as name=/path or /path. Defaults to centrifuge and multidrop.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/astribot_analysis"))
    parser.add_argument("--output-name", default="sample_gripper_z_curves.png")
    parser.add_argument("--manifest-name", default="sample_gripper_z_curves.csv")
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pose-group", default="poses_dict")
    parser.add_argument("--gripper-group", default="poses_dict")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--time-axis",
        choices=["frame", "seconds"],
        default="frame",
        help="Use frame index or elapsed seconds on x axis.",
    )
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot 1 - normalized gripper value, matching the keypose analysis convention.",
    )
    return parser


def elapsed_seconds(h5: h5py.File, n_frames: int) -> np.ndarray:
    if "time" not in h5:
        return np.arange(n_frames, dtype=float)
    time = np.asarray(h5["time"], dtype=float).reshape(-1)
    if time.size < n_frames:
        return np.arange(n_frames, dtype=float)
    return time[:n_frames] - time[0]


def load_episode_curves(
    path: Path,
    arm: str,
    pose_group: str,
    gripper_group: str,
    invert_gripper_value: bool,
    time_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        pose = np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"], dtype=float)
        gripper = prepare_gripper_values(
            np.asarray(h5[f"{gripper_group}/astribot_gripper_{arm}"], dtype=float),
            invert_gripper_value,
        )
        n_frames = min(pose.shape[0], gripper.shape[0])
        x = elapsed_seconds(h5, n_frames) if time_axis == "seconds" else np.arange(n_frames)
    return x, gripper[:n_frames], pose[:n_frames, 2]


def sample_episode_files(task_files: dict[str, list[Path]], num_samples: int, seed: int) -> list[tuple[str, Path]]:
    candidates = [(task, path) for task, files in task_files.items() for path in files]
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    rng = random.Random(seed)
    return rng.sample(candidates, k=min(num_samples, len(candidates)))


def plot_curves(
    samples: list[tuple[str, Path]],
    output_path: Path,
    arm: str,
    pose_group: str,
    gripper_group: str,
    invert_gripper_value: bool,
    time_axis: str,
) -> pd.DataFrame:
    fig_height = max(2.3 * len(samples), 4.0)
    fig, axes = plt.subplots(len(samples), 1, figsize=(11, fig_height), dpi=160, sharex=False)
    axes_arr = np.atleast_1d(axes)
    rows = []
    x_label = "elapsed seconds" if time_axis == "seconds" else "frame"
    gripper_label = "gripper (1 - normalized raw)" if invert_gripper_value else "gripper (normalized raw)"

    for ax, (task, path) in zip(axes_arr, samples):
        x, gripper, z = load_episode_curves(
            path,
            arm=arm,
            pose_group=pose_group,
            gripper_group=gripper_group,
            invert_gripper_value=invert_gripper_value,
            time_axis=time_axis,
        )
        twin = ax.twinx()
        z_line = ax.plot(x, z, color="#1f77b4", linewidth=1.6, label="z")[0]
        gripper_line = twin.plot(
            x, gripper, color="#d62728", linewidth=1.2, alpha=0.9, label="gripper"
        )[0]

        ax.set_ylabel("z", color=z_line.get_color())
        twin.set_ylabel("gripper", color=gripper_line.get_color())
        ax.tick_params(axis="y", labelcolor=z_line.get_color())
        twin.tick_params(axis="y", labelcolor=gripper_line.get_color())
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{task} episode {episode_id(path)}  frames={len(x)}", fontsize=10)
        ax.legend([z_line, gripper_line], ["z", gripper_label], loc="upper right", fontsize=8)
        ax.set_xlabel(x_label)

        rows.append(
            {
                "task": task,
                "episode_id": episode_id(path),
                "file": str(path),
                "frames": len(x),
                "z_min": float(np.nanmin(z)),
                "z_max": float(np.nanmax(z)),
                "gripper_min": float(np.nanmin(gripper)),
                "gripper_max": float(np.nanmax(gripper)),
            }
        )

    fig.suptitle(f"Sampled {arm} gripper and z trajectories", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    args = build_argparser().parse_args()
    task_dirs = parse_task_dirs(args.task_dir) if args.task_dir else DEFAULT_TASK_DIRS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = build_task_files(task_dirs)
    samples = sample_episode_files(task_files, args.num_samples, args.seed)
    manifest = plot_curves(
        samples,
        args.output_dir / args.output_name,
        arm=args.arm,
        pose_group=args.pose_group,
        gripper_group=args.gripper_group,
        invert_gripper_value=args.invert_gripper_value,
        time_axis=args.time_axis,
    )
    manifest_path = args.output_dir / args.manifest_name
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote plot to {args.output_dir / args.output_name}")
    print(f"Wrote sampled episode manifest to {manifest_path}")
    print(manifest[["task", "episode_id", "frames", "z_min", "z_max", "gripper_min", "gripper_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
