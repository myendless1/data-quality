#!/usr/bin/env python3
"""Analyze frame/time distribution for Astribot HDF5 episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from astribot_common import (
    DEFAULT_TASK_DIRS,
    build_task_files,
    frame_summary_table,
    parse_task_dirs,
    plot_frame_histograms,
    summarize_frames,
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
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    task_dirs = parse_task_dirs(args.task_dir) if args.task_dir else DEFAULT_TASK_DIRS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = build_task_files(task_dirs)
    frame_df = summarize_frames(task_files)
    summary_df = frame_summary_table(frame_df)

    frame_df.to_csv(args.output_dir / "frame_stats.csv", index=False)
    summary_df.to_csv(args.output_dir / "frame_summary.csv", index=False)
    plot_frame_histograms(frame_df, args.output_dir / "frame_distribution.png")

    print(f"Wrote frame outputs to {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
