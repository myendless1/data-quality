#!/usr/bin/env python3
"""Export full episode videos and mark selected outlier grasp/place keyframes."""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("outputs/astribot_analysis/keypose_region/selected_keypose_events.csv"),
        help="CSV containing selected outlier events. Used to choose which episodes to export.",
    )
    parser.add_argument(
        "--keypose-stats-csv",
        type=Path,
        default=Path("outputs/astribot_analysis/keypose_stats.csv"),
        help="Full keypose CSV containing grasp/place frames for every episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/astribot_analysis/outliers"),
    )
    parser.add_argument("--camera", default="head")
    parser.add_argument(
        "--keyframe-mark-seconds",
        type=float,
        default=0.75,
        help="Show the keyframe label for this many seconds around each detected keyframe.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=20.0,
        help="FPS used when a valid rate cannot be inferred from the HDF5 time array.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional debug limit. Omit to export every outlier episode.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="imageio/ffmpeg libx264 quality, 0-10 where higher is better/larger.",
    )
    return parser


class Hdf5VideoReader:
    def __init__(self, h5: h5py.File, camera: str):
        self.h5 = h5
        self.camera = camera
        base = f"images_dict/{camera}"
        if f"{base}/rgb" not in h5 or f"{base}/rgb_size" not in h5:
            raise KeyError(f"Camera {camera!r} not found in {Path(h5.filename).name}")
        self.data = h5[f"{base}/rgb"]
        self.sizes = np.asarray(h5[f"{base}/rgb_size"], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.sizes[:-1])))

    @property
    def frame_count(self) -> int:
        return int(self.sizes.shape[0])

    def decode(self, frame: int) -> Image.Image:
        start = int(self.starts[frame])
        end = start + int(self.sizes[frame])
        encoded = np.asarray(self.data[start:end], dtype=np.uint8).tobytes()
        try:
            return Image.open(BytesIO(encoded)).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Failed to decode {self.camera} frame {frame} in {self.h5.filename}"
            ) from exc


def estimate_fps(h5: h5py.File, fallback_fps: float) -> float:
    if "time" in h5 and h5["time"].shape[0] >= 2:
        time = np.asarray(h5["time"], dtype=float)
        duration = float(time[-1] - time[0])
        if np.isfinite(duration) and duration > 0:
            fps = (len(time) - 1) / duration
            if np.isfinite(fps) and 1.0 <= fps <= 120.0:
                return float(fps)
    return float(fallback_fps)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_panel_background(
    image: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha: int = 140,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, alpha))


def event_color(keypose: str) -> tuple[int, int, int]:
    if keypose == "grasp":
        return (20, 120, 255)
    return (255, 150, 30)


def event_summary(events: pd.DataFrame) -> str:
    parts = []
    for _, event in events.sort_values(["frame", "keypose"]).iterrows():
        outlier_suffix = "*" if bool(event.get("is_outlier", False)) else ""
        parts.append(
            f"{event['keypose']}{outlier_suffix}@{int(event['frame'])}"
            f"(x={float(event['x']):.3f},y={float(event['y']):.3f})"
        )
    return "; ".join(parts)


def draw_base_overlay(
    image: Image.Image,
    task: str,
    episode_id: int,
    camera: str,
    frame: int,
    frame_count: int,
    events: pd.DataFrame,
) -> None:
    w, h = image.size
    panel_h = max(78, int(h * 0.12))
    draw_panel_background(image, 0, 0, w, panel_h)
    line1 = f"{task} episode={episode_id} camera={camera} frame={frame}/{frame_count - 1}"
    line2 = f"marked keyframes: {event_summary(events)}  (* in selected outlier region)"
    draw = ImageDraw.Draw(image)
    font1 = load_font(max(22, int(h * 0.035)))
    font2 = load_font(max(18, int(h * 0.028)))
    draw.text((18, 10), line1, font=font1, fill=(255, 255, 255))
    draw.text((18, 44), line2, font=font2, fill=(220, 240, 255))


def active_events_for_frame(events: pd.DataFrame, frame: int, radius_frames: int) -> pd.DataFrame:
    distances = (events["frame"].astype(int) - frame).abs()
    return events.loc[distances <= radius_frames].sort_values(["frame", "keypose"])


def draw_keyframe_overlays(image: Image.Image, active_events: pd.DataFrame, frame: int) -> None:
    if active_events.empty:
        return

    w, h = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    exact = active_events[active_events["frame"].astype(int) == frame]
    border_events = exact if not exact.empty else active_events
    border_width = max(8, w // 120)
    for _, event in border_events.iterrows():
        color = event_color(str(event["keypose"]))
        for inset in range(border_width):
            draw.rectangle(
                (8 + inset, 8 + inset, w - 9 - inset, h - 9 - inset),
                outline=color,
            )

    label_font = load_font(max(64, int(min(w, h) / 6.0)))
    detail_font = load_font(max(24, int(min(w, h) / 18.0)))
    labels = []
    for _, event in active_events.iterrows():
        keypose = str(event["keypose"])
        color = event_color(keypose)
        distance = int(frame - int(event["frame"]))
        if distance == 0:
            status = "KEYFRAME"
        elif distance < 0:
            status = f"{abs(distance)} frames before"
        else:
            status = f"{distance} frames after"
        labels.append(
            {
                "text": keypose,
                "detail": f"{status}  frame={int(event['frame'])}",
                "color": color,
            }
        )

    draw_plain = ImageDraw.Draw(image)
    text_sizes = []
    max_width = 0
    total_height = 0
    gap = 12
    for label in labels:
        text_bbox = draw_plain.textbbox((0, 0), label["text"], font=label_font, stroke_width=5)
        detail_bbox = draw_plain.textbbox((0, 0), label["detail"], font=detail_font)
        block_width = max(text_bbox[2] - text_bbox[0], detail_bbox[2] - detail_bbox[0])
        block_height = (text_bbox[3] - text_bbox[1]) + (detail_bbox[3] - detail_bbox[1]) + 6
        text_sizes.append((block_width, block_height))
        max_width = max(max_width, block_width)
        total_height += block_height
    total_height += gap * (len(labels) - 1)

    pad = 22
    panel_x0 = max(20, (w - max_width) // 2 - pad)
    panel_y0 = max(100, int(h * 0.63) - total_height // 2 - pad)
    panel_x1 = min(w - 20, panel_x0 + max_width + pad * 2)
    panel_y1 = min(h - 20, panel_y0 + total_height + pad * 2)
    draw_panel_background(image, panel_x0, panel_y0, panel_x1, panel_y1, alpha=145)

    y = panel_y0 + pad
    for label, (_, block_height) in zip(labels, text_sizes):
        text = label["text"]
        detail = label["detail"]
        color = label["color"]
        text_bbox = draw_plain.textbbox((0, 0), text, font=label_font, stroke_width=5)
        detail_bbox = draw_plain.textbbox((0, 0), detail, font=detail_font)
        text_width = text_bbox[2] - text_bbox[0]
        detail_width = detail_bbox[2] - detail_bbox[0]
        x_text = (w - text_width) // 2
        x_detail = (w - detail_width) // 2
        draw_plain.text(
            (x_text, y),
            text,
            font=label_font,
            fill=(255, 255, 255),
            stroke_width=7,
            stroke_fill=color,
        )
        y += (text_bbox[3] - text_bbox[1]) + 6
        draw_plain.text((x_detail, y), detail, font=detail_font, fill=(255, 255, 255))
        y += (detail_bbox[3] - detail_bbox[1]) + gap


def safe_output_name(events: pd.DataFrame, camera: str) -> str:
    first = events.iloc[0]
    return f"{first['task']}_episode_{int(first['episode_id']):04d}_outliers_{camera}_full.mp4"


def build_mark_events(selected_events: pd.DataFrame, keypose_stats: pd.DataFrame) -> list[pd.DataFrame]:
    """Return one mark-event DataFrame per outlier episode.

    selected_events chooses which episodes to export. keypose_stats supplies both
    grasp and put frames so every exported full video marks the pair.
    """
    selected = selected_events.copy()
    selected["display_keypose"] = selected["keypose"].replace({"place": "put"})
    selected_files = (
        selected[["task", "episode_id", "file"]]
        .drop_duplicates()
        .sort_values(["task", "episode_id", "file"])
    )
    selected_lookup = {
        (row.file, row.display_keypose)
        for row in selected.itertuples(index=False)
    }

    keypose_by_file = keypose_stats.set_index("file", drop=False)
    groups: list[pd.DataFrame] = []
    for episode in selected_files.itertuples(index=False):
        if episode.file not in keypose_by_file.index:
            raise KeyError(f"Episode missing from keypose stats: {episode.file}")
        stats_row = keypose_by_file.loc[episode.file]
        if isinstance(stats_row, pd.DataFrame):
            stats_row = stats_row.iloc[0]

        rows = []
        for source_name, display_name in [("grasp", "grasp"), ("place", "put")]:
            frame = stats_row[f"{source_name}_frame"]
            if pd.isna(frame):
                continue
            rows.append(
                {
                    "task": stats_row["task"],
                    "episode_id": int(stats_row["episode_id"]),
                    "file": stats_row["file"],
                    "keypose": display_name,
                    "source_keypose": source_name,
                    "frame": int(frame),
                    "x": float(stats_row[f"{source_name}_x"]),
                    "y": float(stats_row[f"{source_name}_y"]),
                    "direction_x": float(stats_row[f"{source_name}_direction_x"]),
                    "direction_y": float(stats_row[f"{source_name}_direction_y"]),
                    "direction_angle_deg": float(stats_row[f"{source_name}_direction_angle_deg"]),
                    "is_outlier": (episode.file, display_name) in selected_lookup,
                }
            )
        if rows:
            groups.append(pd.DataFrame(rows).sort_values(["frame", "keypose"]).reset_index(drop=True))
    return groups


def export_episode_video(
    events: pd.DataFrame,
    output_path: Path,
    camera: str,
    mark_seconds: float,
    fallback_fps: float,
    quality: int,
) -> dict[str, object]:
    first_event = events.iloc[0]
    with h5py.File(first_event["file"], "r") as h5:
        reader = Hdf5VideoReader(h5, camera)
        fps = estimate_fps(h5, fallback_fps)
        radius_frames = max(0, int(round(mark_seconds * fps / 2.0)))

        written_frames = 0
        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=16,
        ) as writer:
            for frame in range(reader.frame_count):
                image = reader.decode(frame)
                draw_base_overlay(
                    image,
                    task=str(first_event["task"]),
                    episode_id=int(first_event["episode_id"]),
                    camera=camera,
                    frame=frame,
                    frame_count=reader.frame_count,
                    events=events,
                )
                active_events = active_events_for_frame(events, frame, radius_frames)
                draw_keyframe_overlays(image, active_events, frame)
                writer.append_data(np.asarray(image))
                written_frames += 1

    return {
        "output_file": str(output_path),
        "camera": camera,
        "task": first_event["task"],
        "episode_id": int(first_event["episode_id"]),
        "source_file": first_event["file"],
        "marked_event_count": int(len(events)),
        "outlier_event_count": int(events["is_outlier"].sum()) if "is_outlier" in events else 0,
        "marked_keyframes": ";".join(
            f"{row['keypose']}:{int(row['frame'])}" for _, row in events.iterrows()
        ),
        "outlier_keyframes": ";".join(
            f"{row['keypose']}:{int(row['frame'])}"
            for _, row in events.iterrows()
            if bool(row.get("is_outlier", False))
        ),
        "frame_count": int(written_frames),
        "fps": round(float(fps), 3),
        "duration_s": round(float(written_frames / fps), 3) if fps > 0 else math.nan,
        "mark_seconds": float(mark_seconds),
        "quality": int(quality),
    }


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_events = pd.read_csv(args.events_csv)
    keypose_stats = pd.read_csv(args.keypose_stats_csv)
    grouped_events = build_mark_events(selected_events, keypose_stats)
    if args.max_videos is not None:
        grouped_events = grouped_events[: args.max_videos]

    manifest_rows = []
    total = len(grouped_events)
    for idx, group in enumerate(grouped_events, start=1):
        output_path = args.output_dir / safe_output_name(group, args.camera)
        row = export_episode_video(
            group,
            output_path=output_path,
            camera=args.camera,
            mark_seconds=args.keyframe_mark_seconds,
            fallback_fps=args.fallback_fps,
            quality=args.quality,
        )
        manifest_rows.append(row)
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] wrote {output_path.name}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / f"outlier_full_video_manifest_{args.camera}.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} full episode videos to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
