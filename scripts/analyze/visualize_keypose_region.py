#!/usr/bin/env python3
"""Visualize keypose frames inside a selected x/y workspace region."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).resolve().parents[1]))

from astribot_common import decode_image_frame


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keypose-csv",
        type=Path,
        default=Path("outputs/astribot_analysis/keypose_stats.csv"),
        help="CSV produced by analyze_keypose_distribution.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/astribot_analysis/keypose_region"),
    )
    parser.add_argument("--x-min", type=float, default=0.15)
    parser.add_argument("--x-max", type=float, default=0.36)
    parser.add_argument("--y-min", type=float, default=-0.45)
    parser.add_argument("--y-max", type=float, default=-0.20)
    parser.add_argument("--camera", default="head")
    parser.add_argument(
        "--max-thumbnails",
        type=int,
        default=200,
        help="Maximum frames per contact sheet; selected evenly if there are more.",
    )
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--cols", type=int, default=7)
    return parser


def normalize_bounds(low: float, high: float) -> tuple[float, float]:
    return (min(low, high), max(low, high))


def selected_events(
    keypose_df: pd.DataFrame,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> pd.DataFrame:
    rows = []
    for keypose in ("grasp", "place"):
        x_col = f"{keypose}_x"
        y_col = f"{keypose}_y"
        frame_col = f"{keypose}_frame"
        dx_col = f"{keypose}_direction_x"
        dy_col = f"{keypose}_direction_y"
        angle_col = f"{keypose}_direction_angle_deg"
        mask = (
            keypose_df[x_col].between(*x_bounds)
            & keypose_df[y_col].between(*y_bounds)
            & keypose_df[frame_col].notna()
        )
        for _, row in keypose_df.loc[mask].iterrows():
            rows.append(
                {
                    "task": row["task"],
                    "episode_id": row["episode_id"],
                    "file": row["file"],
                    "keypose": keypose,
                    "frame": int(row[frame_col]),
                    "x": float(row[x_col]),
                    "y": float(row[y_col]),
                    "direction_x": float(row[dx_col]),
                    "direction_y": float(row[dy_col]),
                    "direction_angle_deg": float(row[angle_col]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "task",
                "episode_id",
                "file",
                "keypose",
                "frame",
                "x",
                "y",
                "direction_x",
                "direction_y",
                "direction_angle_deg",
            ]
        )
    return pd.DataFrame(rows).sort_values(["keypose", "task", "episode_id"]).reset_index(drop=True)


def add_keypose_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    keypose: str,
    color: str,
    label: str,
    alpha: float,
    size: float,
    ray_length: float,
) -> None:
    valid = df[
        [
            f"{keypose}_y",
            f"{keypose}_x",
            f"{keypose}_direction_y",
            f"{keypose}_direction_x",
        ]
    ].dropna()
    if valid.empty:
        return
    ax.scatter(
        valid[f"{keypose}_y"],
        valid[f"{keypose}_x"],
        s=size,
        c=color,
        alpha=alpha,
        label=f"{label} ({len(valid)})",
    )
    ax.quiver(
        valid[f"{keypose}_y"],
        valid[f"{keypose}_x"],
        valid[f"{keypose}_direction_y"] * ray_length,
        valid[f"{keypose}_direction_x"] * ray_length,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0015,
        color=color,
        alpha=min(alpha + 0.15, 1.0),
    )


def plot_region_distribution(
    keypose_df: pd.DataFrame,
    events: pd.DataFrame,
    output_path: Path,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=170)
    colors = {"grasp": "#1f77b4", "place": "#d62728"}

    all_ray_length = math.hypot(0.65, 0.65) * 0.01
    for ax in axes:
        add_keypose_points(ax, keypose_df, "grasp", "#9cc9ec", "all grasp", 0.18, 5, all_ray_length)
        add_keypose_points(ax, keypose_df, "place", "#f1a0a0", "all place", 0.22, 5, all_ray_length)

        rect = Rectangle(
            (y_bounds[0], x_bounds[0]),
            y_bounds[1] - y_bounds[0],
            x_bounds[1] - x_bounds[0],
            fill=False,
            edgecolor="#111111",
            linewidth=1.8,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()

    selected_ray_length = math.hypot(y_bounds[1] - y_bounds[0], x_bounds[1] - x_bounds[0]) * 0.06
    for keypose, group in events.groupby("keypose"):
        axes[0].scatter(
            group["y"],
            group["x"],
            s=30,
            c=colors[keypose],
            edgecolors="white",
            linewidths=0.5,
            label=f"selected {keypose} ({len(group)})",
            zorder=4,
        )
        axes[0].quiver(
            group["y"],
            group["x"],
            group["direction_y"] * selected_ray_length,
            group["direction_x"] * selected_ray_length,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color=colors[keypose],
            zorder=5,
        )
        axes[1].scatter(
            group["y"],
            group["x"],
            s=36,
            c=colors[keypose],
            edgecolors="white",
            linewidths=0.5,
            label=f"selected {keypose} ({len(group)})",
            zorder=4,
        )
        axes[1].quiver(
            group["y"],
            group["x"],
            group["direction_y"] * selected_ray_length,
            group["direction_x"] * selected_ray_length,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color=colors[keypose],
            zorder=5,
        )

    axes[0].set_title(
        f"Selected region in full distribution\n"
        f"y=[{y_bounds[0]:.2f}, {y_bounds[1]:.2f}], x=[{x_bounds[0]:.2f}, {x_bounds[1]:.2f}]"
    )
    axes[1].set_title(f"Zoomed selected region: {len(events)} keyposes")
    axes[1].set_xlim(y_bounds[1] + 0.02, y_bounds[0] - 0.02)
    axes[1].set_ylim(x_bounds[0] - 0.02, x_bounds[1] + 0.02)

    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def resize_rgb(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / float(w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def sample_events(events: pd.DataFrame, max_count: int) -> pd.DataFrame:
    if len(events) <= max_count:
        return events
    indices = np.linspace(0, len(events) - 1, max_count).round().astype(int)
    return events.iloc[np.unique(indices)].reset_index(drop=True)


def make_contact_sheet(
    events: pd.DataFrame,
    output_path: Path,
    camera: str,
    thumb_width: int,
    cols: int,
    title: str,
) -> None:
    if events.empty:
        return

    thumbs: list[tuple[np.ndarray, str]] = []
    for _, event in events.iterrows():
        with h5py.File(event["file"], "r") as h5:
            image = decode_image_frame(h5, camera, int(event["frame"]))
        thumb = resize_rgb(image, thumb_width)
        label = (
            f"{event['keypose']} {event['task']} ep={int(event['episode_id'])} f={int(event['frame'])}\n"
            f"x={event['x']:.3f}, y={event['y']:.3f}, dir={event['direction_angle_deg']:.1f} deg"
        )
        thumbs.append((thumb, label))

    rows = math.ceil(len(thumbs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.45, rows * 2.15), dpi=150)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (thumb, label) in zip(axes_arr, thumbs):
        ax.imshow(thumb)
        ax.set_title(label, fontsize=6.2)
        ax.axis("off")
    for ax in axes_arr[len(thumbs) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_bounds = normalize_bounds(args.x_min, args.x_max)
    y_bounds = normalize_bounds(args.y_min, args.y_max)
    keypose_df = pd.read_csv(args.keypose_csv)
    events = selected_events(keypose_df, x_bounds, y_bounds)

    events_path = args.output_dir / "selected_keypose_events.csv"
    events.to_csv(events_path, index=False)

    plot_region_distribution(
        keypose_df,
        events,
        args.output_dir / "keypose_region_distribution.png",
        x_bounds,
        y_bounds,
    )

    for keypose in ("grasp", "place"):
        subset = sample_events(events[events["keypose"] == keypose].reset_index(drop=True), args.max_thumbnails)
        make_contact_sheet(
            subset,
            args.output_dir / f"keypose_region_{keypose}_{args.camera}_contact_sheet.png",
            args.camera,
            args.thumb_width,
            args.cols,
            f"{keypose} keyframes in selected region ({len(subset)}/{(events['keypose'] == keypose).sum()})",
        )

    make_contact_sheet(
        sample_events(events.reset_index(drop=True), args.max_thumbnails),
        args.output_dir / f"keypose_region_all_{args.camera}_contact_sheet.png",
        args.camera,
        args.thumb_width,
        args.cols,
        f"All keyframes in selected region ({min(len(events), args.max_thumbnails)}/{len(events)})",
    )

    print(f"Wrote {len(events)} selected keyposes to {events_path}")
    if not events.empty:
        print(events.groupby(["keypose", "task"]).size().to_string())
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
