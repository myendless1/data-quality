#!/usr/bin/env python3
"""Analyze grasp/place xy and projected rotation distributions."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from astribot_common import (
    DEFAULT_TASK_DIRS,
    build_task_files,
    collect_keypose_records,
    compute_xy_limits,
    episode_id,
    keypose_plot_points_from_df,
    parse_task_dirs,
    plot_gripper_interval_stats,
    plot_keypose_distribution,
    plot_multi_episode_debug,
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
        help="Centered moving-average window, in frames, before finding the last negative z slope.",
    )
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - normalized gripper value before thresholding.",
    )
    parser.add_argument(
        "--direction-axis",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="-y",
        help="Tool-frame axis projected to the world xy plane.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=["column", "row"],
        default="column",
        help="Use rotation matrix columns or rows for the projected axis.",
    )
    parser.add_argument("--debug-camera", default="head")
    parser.add_argument("--debug-task", default="multidrop")
    parser.add_argument(
        "--debug-index", type=int, default=0, help="Index into sorted files for debug figure."
    )
    parser.add_argument("--debug-count", type=int, default=4)
    parser.add_argument(
        "--debug-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample debug episodes. Use --no-debug-random for sequential selection.",
    )
    parser.add_argument("--debug-seed", type=int, default=20260622)
    parser.add_argument("--skip-debug", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    task_dirs = parse_task_dirs(args.task_dir) if args.task_dir else DEFAULT_TASK_DIRS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = build_task_files(task_dirs)
    keypose_df = collect_keypose_records(
        task_files,
        arm=args.arm,
        pose_group=args.pose_group,
        gripper_group=args.gripper_group,
        low_gripper_threshold=args.low_gripper_threshold,
        z_smooth_window=args.z_smooth_window,
        invert_gripper_value=args.invert_gripper_value,
        direction_axis=args.direction_axis,
        direction_mode=args.direction_mode,
    )
    keypose_df.to_csv(args.output_dir / "keypose_stats.csv", index=False)

    all_limits = compute_xy_limits(keypose_plot_points_from_df(keypose_df))
    plot_keypose_distribution(
        keypose_df,
        args.output_dir / "keypose_distribution_all.png",
        "Keypose xy/direction distribution",
        all_limits,
    )
    for task, group in keypose_df.groupby("task"):
        plot_keypose_distribution(
            group,
            args.output_dir / f"keypose_distribution_{task}.png",
            f"{task} keypose xy/direction distribution",
            all_limits,
        )
    plot_gripper_interval_stats(
        keypose_df,
        args.output_dir / "low_gripper_interval_mean_var_distribution.png",
    )

    if not args.skip_debug:
        if args.debug_task not in task_files:
            raise KeyError(f"Unknown debug task {args.debug_task}; options: {sorted(task_files)}")
        candidates = task_files[args.debug_task]
        if args.debug_random:
            rng = random.Random(args.debug_seed)
            debug_files = rng.sample(candidates, k=min(args.debug_count, len(candidates)))
        else:
            debug_files = candidates[args.debug_index : args.debug_index + args.debug_count]
        if not debug_files:
            raise IndexError(f"No debug episodes from index {args.debug_index}")
        plot_multi_episode_debug(
            debug_files,
            args.output_dir / f"debug_{args.debug_task}_episode_{episode_id(debug_files[0])}.png",
            task=args.debug_task,
            arm=args.arm,
            camera=args.debug_camera,
            pose_group=args.pose_group,
            gripper_group=args.gripper_group,
            low_gripper_threshold=args.low_gripper_threshold,
            z_smooth_window=args.z_smooth_window,
            invert_gripper_value=args.invert_gripper_value,
            direction_axis=args.direction_axis,
            direction_mode=args.direction_mode,
        )

    missing_grasp = int(keypose_df["grasp_frame"].isna().sum())
    missing_place = int(keypose_df["place_frame"].isna().sum())
    print(f"Wrote keypose outputs to {args.output_dir}")
    print(f"Episodes: {len(keypose_df)}; missing grasp={missing_grasp}; missing place={missing_place}")
    print(
        f"Gripper value: {'1 - normalized raw' if args.invert_gripper_value else 'normalized raw'}"
    )
    print(
        "Keyframe rule: "
        f"longest gripper<{args.low_gripper_threshold}; "
        f"grasp=start; place=last negative smoothed-z slope; "
        f"z_smooth_window={args.z_smooth_window}"
    )
    print(f"Direction projection: axis={args.direction_axis}, mode={args.direction_mode}")


if __name__ == "__main__":
    main()
