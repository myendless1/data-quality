#!/usr/bin/env python3
"""Filter valid Astribot HDF5 episodes by data-quality rules."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help=(
            "Dataset directory containing .hdf5 files. Can be passed multiple times. "
            "Use name=/path or /path; the name is only used in the manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/astribot_analysis/valid_hdf5_files.txt"),
        help="Text file containing one valid HDF5 path per line.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/astribot_analysis/hdf5_filter_manifest.csv"),
        help="CSV manifest containing valid/invalid status and rule metrics.",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--gripper-group", default="poses_dict")
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - normalized gripper value before thresholding. Default gives 0=closed, 1=open.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.105,
        help="Count frames whose processed gripper value is below this threshold.",
    )
    parser.add_argument(
        "--min-frames-below-threshold",
        type=int,
        default=100,
        help=(
            "Rule 1: mark an episode invalid if it has fewer than this many frames with "
            "processed gripper < --gripper-threshold."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .hdf5 files recursively under each dataset directory.",
    )
    return parser


def parse_dataset_dirs(values: Iterable[str]) -> dict[str, Path]:
    dataset_dirs: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            dataset_dirs[name] = Path(path)
        else:
            path = Path(value)
            dataset_dirs[path.name] = path
    return dataset_dirs


def episode_id(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def sort_key(path: Path) -> tuple[int, int | str]:
    episode = episode_id(path)
    if episode is None:
        return (1, path.stem)
    return (0, episode)


def list_hdf5_files(dataset_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    return sorted(dataset_dir.glob(pattern), key=sort_key)


def prepare_gripper_values(raw_gripper: np.ndarray, invert_gripper_value: bool) -> np.ndarray:
    """Convert gripper readings to the same 0..1 convention used by the analysis scripts."""
    gripper = np.asarray(raw_gripper, dtype=float).reshape(-1)
    finite = gripper[np.isfinite(gripper)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.5:
        gripper = gripper / 100.0
    if invert_gripper_value:
        gripper = 1.0 - gripper
    return gripper


def filter_episode(
    path: Path,
    task: str,
    arm: str,
    gripper_group: str,
    invert_gripper_value: bool,
    gripper_threshold: float,
    min_frames_below_threshold: int,
) -> dict[str, object]:
    dataset_key = f"{gripper_group}/astribot_gripper_{arm}"
    with h5py.File(path, "r") as h5:
        if dataset_key not in h5:
            return {
                "task": task,
                "episode_id": episode_id(path),
                "file": str(path),
                "valid": False,
                "invalid_reason": f"missing_dataset:{dataset_key}",
                "frames": 0,
                "finite_frames": 0,
                "gripper_below_threshold_frames": 0,
                "gripper_below_threshold_ratio": "",
                "gripper_min": "",
                "gripper_max": "",
                "gripper_mean": "",
            }
        raw_gripper = np.asarray(h5[dataset_key], dtype=float)

    gripper = prepare_gripper_values(raw_gripper, invert_gripper_value)
    finite_mask = np.isfinite(gripper)
    below_count = int(np.sum(finite_mask & (gripper < gripper_threshold)))
    finite_count = int(np.sum(finite_mask))
    invalid_reasons: list[str] = []

    if below_count < min_frames_below_threshold:
        invalid_reasons.append(
            f"rule1_gripper_lt_{gripper_threshold:g}_frames_{below_count}"
            f"_less_than_{min_frames_below_threshold}"
        )

    valid = not invalid_reasons
    return {
        "task": task,
        "episode_id": episode_id(path),
        "file": str(path),
        "valid": valid,
        "invalid_reason": ";".join(invalid_reasons),
        "frames": int(gripper.size),
        "finite_frames": finite_count,
        "gripper_below_threshold_frames": below_count,
        "gripper_below_threshold_ratio": below_count / finite_count if finite_count else "",
        "gripper_min": float(np.nanmin(gripper)) if finite_count else "",
        "gripper_max": float(np.nanmax(gripper)) if finite_count else "",
        "gripper_mean": float(np.nanmean(gripper)) if finite_count else "",
    }


def filter_by_gripper(
    dataset_dirs: dict[str, Path],
    output: Path,
    manifest: Path,
    arm: str = "right",
    gripper_group: str = "poses_dict",
    invert_gripper_value: bool = True,
    gripper_threshold: float = 0.105,
    min_frames_below_threshold: int = 100,
    recursive: bool = False,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Step 1: keep HDF5 episodes with enough closed-gripper frames."""
    if min_frames_below_threshold < 0:
        raise ValueError("min_frames_below_threshold must be non-negative")

    rows: list[dict[str, object]] = []
    for task, dataset_dir in dataset_dirs.items():
        files = list_hdf5_files(dataset_dir, recursive)
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found for {task}: {dataset_dir}")
        for path in files:
            rows.append(
                filter_episode(
                    path=path,
                    task=task,
                    arm=arm,
                    gripper_group=gripper_group,
                    invert_gripper_value=invert_gripper_value,
                    gripper_threshold=gripper_threshold,
                    min_frames_below_threshold=min_frames_below_threshold,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    valid_rows = [row for row in rows if row["valid"]]
    output.write_text("\n".join(str(row["file"]) for row in valid_rows) + "\n")

    fieldnames = [
        "task",
        "episode_id",
        "valid",
        "invalid_reason",
        "frames",
        "finite_frames",
        "gripper_below_threshold_frames",
        "gripper_below_threshold_ratio",
        "gripper_min",
        "gripper_max",
        "gripper_mean",
        "file",
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return [Path(str(row["file"])) for row in valid_rows], rows


def main() -> None:
    args = build_argparser().parse_args()
    valid_paths, rows = filter_by_gripper(
        dataset_dirs=parse_dataset_dirs(args.dataset_dir),
        output=args.output,
        manifest=args.manifest,
        arm=args.arm,
        gripper_group=args.gripper_group,
        invert_gripper_value=args.invert_gripper_value,
        gripper_threshold=args.gripper_threshold,
        min_frames_below_threshold=args.min_frames_below_threshold,
        recursive=args.recursive,
    )

    invalid_count = len(rows) - len(valid_paths)
    print(f"Wrote {len(valid_paths)} valid HDF5 paths to {args.output}")
    print(f"Wrote filter manifest for {len(rows)} episodes to {args.manifest}")
    print(f"valid={len(valid_rows)} invalid={invalid_count}")


if __name__ == "__main__":
    main()
