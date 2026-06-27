#!/usr/bin/env python3
"""Run the formal Astribot HDF5 filtering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from step1_filter_valid_hdf5_by_gripper import filter_by_gripper, parse_dataset_dirs
from step2_sample_by_grasp_place_fps import filter_by_grasp_place_fps
from visualize_projected_sample_distribution import (
    DEFAULT_BACKGROUND,
    DEFAULT_MODEL,
    visualize_projected_distribution,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help=(
            "Dataset directory containing .hdf5 files. Can be passed multiple times. "
            "Use name=/path or /path; the name is only used in manifests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/astribot_filter"),
        help="Directory for all step outputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of HDF5 episodes to keep per task after FPS sampling.",
    )
    parser.add_argument(
        "--output-path-format",
        choices=["path", "name"],
        default="path",
        help="Write full HDF5 paths or only HDF5 file names to the final txt.",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pose-group", default="poses_dict")
    parser.add_argument("--gripper-group", default="poses_dict")
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - normalized gripper value before thresholding.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.105,
        help="Step1 threshold for closed-gripper frame counting.",
    )
    parser.add_argument(
        "--min-frames-below-threshold",
        type=int,
        default=100,
        help="Step1 minimum number of frames with gripper below threshold.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Step1 searches recursively under each dataset directory.",
    )
    parser.add_argument(
        "--low-gripper-threshold",
        type=float,
        default=0.11,
        help="Step2 threshold used to find grasp/place keyframes.",
    )
    parser.add_argument(
        "--z-smooth-window",
        type=int,
        default=15,
        help="Step2 centered moving-average window before finding place frame.",
    )
    parser.add_argument(
        "--fps-dims",
        choices=["xy", "xyz"],
        default="xyz",
        help="Use xy or xyz coordinates for FPS distances.",
    )
    parser.add_argument(
        "--initial-sample",
        choices=["center-farthest", "random"],
        default="center-farthest",
        help="How step2 chooses the first selected episode.",
    )
    parser.add_argument(
        "--sampling-scope",
        choices=["per-task", "global"],
        default="per-task",
        help="Sample --num-samples per task, or globally across all tasks.",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument(
        "--projection-model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Calibration model used for final grasp/place projection distribution plots.",
    )
    parser.add_argument(
        "--projection-output-dir",
        type=Path,
        default=None,
        help="Directory for projection distribution plots. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--projection-background",
        type=Path,
        default=DEFAULT_BACKGROUND,
        help="Background image for projection distribution plots.",
    )
    parser.add_argument(
        "--projection-group-by",
        choices=["task", "keypose"],
        default="task",
        help="Create projection plots grouped by task or keypose.",
    )
    parser.add_argument("--skip-visualization", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step1_output = args.output_dir / "step1_valid_hdf5_files.txt"
    step1_manifest = args.output_dir / "step1_hdf5_filter_manifest.csv"
    step2_output = args.output_dir / "step2_fps_sampled_hdf5_files.txt"
    step2_manifest = args.output_dir / "step2_fps_sampling_manifest.csv"

    valid_paths, step1_rows = filter_by_gripper(
        dataset_dirs=parse_dataset_dirs(args.dataset_dir),
        output=step1_output,
        manifest=step1_manifest,
        arm=args.arm,
        gripper_group=args.gripper_group,
        invert_gripper_value=args.invert_gripper_value,
        gripper_threshold=args.gripper_threshold,
        min_frames_below_threshold=args.min_frames_below_threshold,
        recursive=args.recursive,
    )
    selected_paths, step2_df = filter_by_grasp_place_fps(
        input_path=step1_output,
        output=step2_output,
        manifest=step2_manifest,
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

    projection_outputs = []
    if not args.skip_visualization:
        projection_output_dir = (
            args.projection_output_dir
            if args.projection_output_dir is not None
            else args.output_dir / "projection_distribution"
        )
        projection_outputs = visualize_projected_distribution(
            manifest=step2_manifest,
            model_path=args.projection_model,
            output_dir=projection_output_dir,
            arm=args.arm,
            pose_group=args.pose_group,
            background=args.projection_background,
            group_by=args.projection_group_by,
        )

    invalid_count = len(step1_rows) - len(valid_paths)
    print(f"Step1: valid={len(valid_paths)} invalid={invalid_count}")
    print(f"Step1 outputs: {step1_output}, {step1_manifest}")
    print(f"Step2: selected={len(selected_paths)} finite_candidates={len(step2_df)}")
    print(f"Step2 outputs: {step2_output}, {step2_manifest}")
    if projection_outputs:
        print("Projection distribution outputs:")
        for path in projection_outputs:
            print(path)


if __name__ == "__main__":
    main()
