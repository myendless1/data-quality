#!/usr/bin/env python3
"""Serve a sampling-guidance UI over live projected grasp/place distributions."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import random
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import h5py
import numpy as np


TASKS = ("centrifuge", "multidrop")
KEYPOSES = ("grasp", "place")


@dataclass(frozen=True)
class ProjectionModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rvec: list[float]
    tvec: list[float]
    offsets: dict[str, object]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/media/damoxing/datasets/astribot_tasks/myendless"),
        help="Directory containing task subdirs like centrifuge/ and multidrop/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/calibration_bbox_labeled/camera_model_shared_tool.json"),
        help="Projection model fitted from the shared-tool labels.",
    )
    parser.add_argument("--background", type=Path, default=Path("image.png"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--candidate-grid", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pose-group", default="poses_dict")
    parser.add_argument("--gripper-group", default="poses_dict")
    parser.add_argument("--low-gripper-threshold", type=float, default=0.11)
    parser.add_argument("--z-smooth-window", type=int, default=15)
    parser.add_argument(
        "--invert-gripper-value",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--direction-axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="-y")
    parser.add_argument("--direction-mode", choices=["column", "row"], default="column")
    return parser


def load_model(path: Path) -> ProjectionModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    intr = data["intrinsics"]
    extr = data["extrinsics"]
    return ProjectionModel(
        width=int(data.get("width", 1280)),
        height=int(data.get("height", 720)),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        rvec=[float(v) for v in extr["rotation_vector"]],
        tvec=[float(v) for v in extr["translation"]],
        offsets=data.get("object_offsets", data.get("object_offsets_world", {})),
    )


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


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


def project_points(points_world: np.ndarray, model: ProjectionModel) -> tuple[np.ndarray, np.ndarray]:
    rotation = rodrigues(np.asarray(model.rvec, dtype=float))
    points_cam = (rotation @ points_world.T).T + np.asarray(model.tvec, dtype=float).reshape(1, 3)
    z = points_cam[:, 2]
    uv = np.empty((points_world.shape[0], 2), dtype=float)
    uv[:, 0] = model.fx * points_cam[:, 0] / z + model.cx
    uv[:, 1] = model.fy * points_cam[:, 1] / z + model.cy
    return uv, z


def apply_model_offset(model: ProjectionModel, point_world: np.ndarray, rotation_world_tool: np.ndarray) -> np.ndarray:
    offsets = model.offsets or {}
    if "shared_tool_frame" not in offsets:
        return point_world
    offset_tool = np.asarray(offsets["shared_tool_frame"], dtype=float)
    return point_world + rotation_world_tool @ offset_tool


def episode_id(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def list_episode_files(input_dir: Path, task: str) -> list[Path]:
    task_dir = input_dir / task
    files = list(task_dir.glob("*.hdf5"))
    return sorted(files, key=lambda p: episode_id(p) if episode_id(p) is not None else p.stem)


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
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_keypose_frames(
    gripper: np.ndarray,
    z: np.ndarray,
    low_gripper_threshold: float,
    z_smooth_window: int,
) -> dict[str, int | None]:
    n_frames = min(len(gripper), len(z))
    gripper = np.asarray(gripper[:n_frames], dtype=float).reshape(-1)
    z = np.asarray(z[:n_frames], dtype=float).reshape(-1)
    interval = longest_true_interval(np.isfinite(gripper) & (gripper < low_gripper_threshold))
    if interval is None:
        return {"grasp": None, "place": None}
    start, end = interval
    smoothed_z = smooth_centered(z[start : end + 1], z_smooth_window)
    if smoothed_z.size < 2:
        return {"grasp": int(start), "place": None}
    slopes = np.gradient(smoothed_z)
    negative = np.flatnonzero(np.isfinite(slopes) & (slopes < 0))
    place = start + int(negative[-1]) if negative.size else None
    return {"grasp": int(start), "place": int(place) if place is not None else None}


def direction_from_pose(
    pose: np.ndarray,
    direction_axis: str,
    direction_mode: str,
) -> tuple[float | None, float | None]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[direction_axis[1]]
    sign = 1.0 if direction_axis[0] == "+" else -1.0
    rotation = rotation_matrix_from_xyzw(pose[3:7])
    basis = rotation if direction_mode == "column" else rotation.T
    vector = sign * basis[:, axis_idx]
    xy = np.asarray(vector[:2], dtype=float)
    norm = float(np.linalg.norm(xy))
    if norm == 0 or not np.isfinite(norm):
        return None, None
    dx, dy = xy / norm
    return float(dx), float(dy)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            cross_x = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def nearest_distances(candidates: np.ndarray, points: np.ndarray) -> np.ndarray:
    diff = candidates[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


class SamplingState:
    def __init__(
        self,
        input_dir: Path,
        model_path: Path,
        background: Path,
        candidate_grid: int,
        top_k: int,
        seed: int | None,
        arm: str,
        pose_group: str,
        gripper_group: str,
        low_gripper_threshold: float,
        z_smooth_window: int,
        invert_gripper_value: bool,
        direction_axis: str,
        direction_mode: str,
    ) -> None:
        self.input_dir = input_dir
        self.model_path = model_path
        self.background = background
        self.candidate_grid = candidate_grid
        self.top_k = top_k
        self.rng = random.Random(seed)
        self.arm = arm
        self.pose_group = pose_group
        self.gripper_group = gripper_group
        self.low_gripper_threshold = low_gripper_threshold
        self.z_smooth_window = z_smooth_window
        self.invert_gripper_value = invert_gripper_value
        self.direction_axis = direction_axis
        self.direction_mode = direction_mode
        self.model = load_model(model_path)
        self.file_cache: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        self.last_scan_error: str | None = None
        self.last_refresh_at = -1e12
        self.refresh_interval_s = 10.0
        self.groups_cache: dict[str, dict[str, object]] = {}
        self.groups_cache_key: tuple[tuple[str, object], ...] | None = None

    def refresh(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now - self.last_refresh_at < self.refresh_interval_s:
                return
            self.last_refresh_at = now
            current_paths: set[str] = set()
            try:
                changed = False
                for task in TASKS:
                    for path in list_episode_files(self.input_dir, task):
                        current_paths.add(str(path))
                        stat = path.stat()
                        cached = self.file_cache.get(str(path))
                        signature = (stat.st_mtime_ns, stat.st_size)
                        if cached and cached.get("signature") == signature:
                            continue
                        points = self.project_episode(task, path)
                        self.file_cache[str(path)] = {
                            "task": task,
                            "episode_id": episode_id(path),
                            "signature": signature,
                            "points": points,
                        }
                        changed = True
                for cached_path in list(self.file_cache):
                    if cached_path not in current_paths:
                        del self.file_cache[cached_path]
                        changed = True
                if changed:
                    self.groups_cache_key = None
                self.last_scan_error = None
            except Exception as exc:
                self.last_scan_error = f"{type(exc).__name__}: {exc}"

    def project_episode(self, task: str, path: Path) -> list[dict[str, object]]:
        with h5py.File(path, "r") as h5:
            raw_gripper = np.asarray(h5[f"{self.gripper_group}/astribot_gripper_{self.arm}"])
            poses = np.asarray(h5[f"{self.pose_group}/astribot_arm_{self.arm}"], dtype=float)
        gripper = prepare_gripper_values(raw_gripper, self.invert_gripper_value)
        frames = detect_keypose_frames(
            gripper,
            poses[:, 2],
            self.low_gripper_threshold,
            self.z_smooth_window,
        )
        rows: list[dict[str, object]] = []
        for keypose, frame in frames.items():
            if frame is None or frame < 0 or frame >= poses.shape[0]:
                continue
            pose = poses[int(frame)]
            point = np.asarray(pose[:3], dtype=float)
            rotation = rotation_matrix_from_xyzw(pose[3:7])
            shifted = apply_model_offset(self.model, point, rotation).reshape(1, 3)
            uv, depth = project_points(shifted, self.model)
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if depth[0] <= 0 or not (0 <= u <= self.model.width and 0 <= v <= self.model.height):
                continue
            direction_u = None
            direction_v = None
            dx, dy = direction_from_pose(pose, self.direction_axis, self.direction_mode)
            if dx is not None and dy is not None:
                end_world = shifted + np.asarray([[dx, dy, 0.0]], dtype=float) * 0.04
                end_uv, end_depth = project_points(end_world, self.model)
                if end_depth[0] > 0:
                    direction_u = float(end_uv[0, 0] - u)
                    direction_v = float(end_uv[0, 1] - v)
            rows.append(
                {
                    "task": task,
                    "keypose": keypose,
                    "episode_id": episode_id(path),
                    "frame": int(frame),
                    "file": str(path),
                    "x": u,
                    "y": v,
                    "direction_u": direction_u,
                    "direction_v": direction_v,
                }
            )
        return rows

    def all_points_unlocked(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for item in self.file_cache.values():
            rows.extend(item.get("points", []))  # type: ignore[arg-type]
        return rows

    def cache_key_unlocked(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            sorted((path, item.get("signature")) for path, item in self.file_cache.items())
        )

    def build_groups(self, points: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        groups: dict[str, dict[str, object]] = {}
        for task in TASKS:
            for keypose in KEYPOSES:
                name = f"{task}:{keypose}"
                pts = np.asarray(
                    [[p["x"], p["y"]] for p in points if p["task"] == task and p["keypose"] == keypose],
                    dtype=float,
                )
                candidates: list[dict[str, float]] = []
                hull = convex_hull(pts) if len(pts) >= 3 else np.empty((0, 2), dtype=float)
                if len(hull) >= 3:
                    min_x, min_y = hull.min(axis=0)
                    max_x, max_y = hull.max(axis=0)
                    xs = np.linspace(min_x, max_x, self.candidate_grid)
                    ys = np.linspace(min_y, max_y, self.candidate_grid)
                    grid = np.asarray(
                        [
                            (float(x), float(y))
                            for y in ys
                            for x in xs
                            if point_in_polygon((float(x), float(y)), hull)
                        ],
                        dtype=float,
                    )
                    if len(grid):
                        dist = nearest_distances(grid, pts)
                        order = np.argsort(-dist)[: self.top_k]
                        candidates = [
                            {"x": float(grid[i, 0]), "y": float(grid[i, 1]), "score": float(dist[i])}
                            for i in order
                        ]
                groups[name] = {
                    "task": task,
                    "keypose": keypose,
                    "count": int(len(pts)),
                    "hull": hull.tolist(),
                    "top_candidates": candidates,
                }
        return groups

    def sample_pair(self, task: str, groups: dict[str, dict[str, object]]) -> dict[str, object]:
        samples: dict[str, object] = {}
        for keypose in KEYPOSES:
            name = f"{task}:{keypose}"
            group = groups.get(name, {})
            candidates = group.get("top_candidates") or []
            if not candidates:
                samples[keypose] = {"error": "no candidates available", "group": name}
                continue
            samples[keypose] = {
                "group": name,
                "task": task,
                "keypose": keypose,
                "candidate": self.rng.choice(candidates),  # type: ignore[arg-type]
                "top_candidates": candidates,
                "count": group.get("count", 0),
            }
        return samples

    def payload(self, query: dict[str, list[str]]) -> dict[str, object]:
        raw_task = query.get("task", ["centrifuge"])[0]
        task = raw_task if raw_task in TASKS else "centrifuge"
        self.refresh()
        with self.lock:
            points = self.all_points_unlocked()
            cache_key = self.cache_key_unlocked()
            if self.groups_cache_key == cache_key:
                groups = self.groups_cache
            else:
                groups = self.build_groups(points)
                self.groups_cache = groups
                self.groups_cache_key = cache_key
        return {
            "width": self.model.width,
            "height": self.model.height,
            "task": task,
            "samples": self.sample_pair(task, groups),
            "groups": groups,
            "points": points,
            "cache": {
                "files": len(self.file_cache),
                "points": len(points),
                "error": self.last_scan_error,
                "input_dir": str(self.input_dir),
                "model": str(self.model_path),
            },
        }

    def background_data_url(self) -> str:
        mime = mimetypes.guess_type(str(self.background))[0] or "image/png"
        data = base64.b64encode(self.background.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"


class GuidanceHandler(BaseHTTPRequestHandler):
    state: SamplingState
    ui_path: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self.serve_file(self.ui_path)
            return
        if path == "/api/data":
            self.send_json(self.state.payload(parse_qs(parsed.query)))
            return
        if path == "/api/background":
            self.send_json({"data_url": self.state.background_data_url()})
            return
        self.send_error(404)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    args = build_argparser().parse_args()
    state = SamplingState(
        input_dir=args.input_dir,
        model_path=args.model,
        background=args.background,
        candidate_grid=args.candidate_grid,
        top_k=args.top_k,
        seed=args.seed,
        arm=args.arm,
        pose_group=args.pose_group,
        gripper_group=args.gripper_group,
        low_gripper_threshold=args.low_gripper_threshold,
        z_smooth_window=args.z_smooth_window,
        invert_gripper_value=args.invert_gripper_value,
        direction_axis=args.direction_axis,
        direction_mode=args.direction_mode,
    )
    GuidanceHandler.state = state
    GuidanceHandler.ui_path = Path(__file__).with_name("sampling_guidance_ui.html").resolve()
    server = ThreadingHTTPServer((args.host, args.port), GuidanceHandler)
    print(f"Open http://{args.host}:{args.port}")
    print("Data will be scanned on the first /api/data request, then refreshed at most every 10s.")
    server.serve_forever()


if __name__ == "__main__":
    main()
