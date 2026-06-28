#!/usr/bin/env python3
"""Serve a sampling-guidance UI over live projected grasp/place distributions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import h5py
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation


TASKS = ("centrifuge", "multidrop")
KEYPOSES = ("grasp", "place")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "calibration" / "camera_model_shared_tool.json"
DEFAULT_BACKGROUND_PATH = SCRIPT_DIR / "image.png"
DEFAULT_ALIGNMENT_PATH = SCRIPT_DIR / "calibration" / "urdf_fk_alignment_right_arm.json"
DEFAULT_URDF_PATH = (
    SCRIPT_DIR
    / "astribot_descriptions"
    / "urdf"
    / "astribot_s1_urdf"
    / "astribot_whole_body.urdf"
)
DEFAULT_RIGHT_ARM_URDF_PATH = DEFAULT_URDF_PATH.with_name("astribot_arm_right.urdf")
DEFAULT_GRASP_HEIGHT_M = 0.7821216622438248
DEFAULT_CENTRIFUGE_PLACE_HEIGHT_M = 0.8658671177760737
DEFAULT_MULTIDROP_PLACE_HEIGHT_M = 0.7886867696617663
DEFAULT_SAMPLE_CACHE_PATH = SCRIPT_DIR / "sampling_guidance_pair_cache.json"
DEFAULT_COLLECTION_ROOT = SCRIPT_DIR / "collection_inbox"
COLLECTION_FILE_SUFFIXES = {".hdf5", ".h5"}
FIXED_DATASET_JOINT_VALUES = {
    0: -0.019,
    1: -0.008,
    2: -0.086,
    3: 0.594,
    4: -1.192,
    5: 0.597,
    6: 0.002,
    23: -0.009,
    24: 0.854,
}
FIXED_DATASET_JOINT_NAMES = {
    0: "dataset_joint_0",
    1: "dataset_joint_1",
    2: "dataset_joint_2",
    3: "astribot_torso_joint_1",
    4: "astribot_torso_joint_2",
    5: "astribot_torso_joint_3",
    6: "astribot_torso_joint_4",
    23: "astribot_head_joint_1",
    24: "astribot_head_joint_2",
}
RIGHT_ARM_EE_OFFSET_TOOL = np.asarray([0.0, -0.15, 0.0], dtype=float)


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
        default=DEFAULT_MODEL_PATH,
        help="Projection model fitted from the shared-tool labels.",
    )
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--right-arm-urdf", type=Path, default=DEFAULT_RIGHT_ARM_URDF_PATH)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--candidate-grid", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sample-attempts", type=int, default=60000)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE_PATH)
    parser.add_argument("--precompute-count", type=int, default=100)
    parser.add_argument(
        "--centrifuge-collection-dir",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT / "centrifuge",
    )
    parser.add_argument(
        "--multidrop-collection-dir",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT / "multidrop",
    )
    parser.add_argument("--collection-position-tolerance-m", type=float, default=0.03)
    parser.add_argument("--collection-rotation-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--collection-scan-interval-s", type=float, default=0.8)
    parser.add_argument("--collection-file-stable-s", type=float, default=1.0)
    parser.add_argument("--collection-read-timeout-s", type=float, default=10.0)
    parser.add_argument("--workspace-xy", default="0.18,0.65,-0.45,0.20")
    parser.add_argument("--grasp-height", type=float, default=DEFAULT_GRASP_HEIGHT_M)
    parser.add_argument("--centrifuge-place-height", type=float, default=DEFAULT_CENTRIFUGE_PLACE_HEIGHT_M)
    parser.add_argument("--multidrop-place-height", type=float, default=DEFAULT_MULTIDROP_PLACE_HEIGHT_M)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.01)
    parser.add_argument("--ik-rotation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--orientation-neighbors", type=int, default=24)
    parser.add_argument("--object-length-cm", type=float, default=13.0)
    parser.add_argument("--object-width-cm", type=float, default=8.0)
    parser.add_argument("--block-corridor-cm", type=float, default=7.0)
    parser.add_argument(
        "--min-tool-direction-dot",
        type=float,
        default=0.0,
        help="Minimum XY dot(tool direction, shoulder-to-target direction). Use -1 to disable.",
    )
    parser.add_argument(
        "--right-shoulder-xy",
        default="0.0,-0.22",
        help="Right shoulder xy in the same world frame as poses_dict, formatted as x,y.",
    )
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


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(value, tmp, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def rng_state_to_json(state: object) -> object:
    if isinstance(state, tuple):
        return [rng_state_to_json(item) for item in state]
    return state


def rng_state_from_json(state: object) -> object:
    if isinstance(state, list):
        return tuple(rng_state_from_json(item) for item in state)
    return state


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


def list_collection_files(collection_dir: Path) -> list[Path]:
    if not collection_dir.exists():
        return []
    return sorted(
        (
            path
            for path in collection_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in COLLECTION_FILE_SUFFIXES
        ),
        key=lambda p: str(p),
    )


def quat_error_deg(actual_xyzw: np.ndarray, expected_xyzw: np.ndarray) -> float:
    actual = np.asarray(actual_xyzw, dtype=float).reshape(4)
    expected = np.asarray(expected_xyzw, dtype=float).reshape(4)
    return float(math.degrees((Rotation.from_quat(actual).inv() * Rotation.from_quat(expected)).magnitude()))


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


def point_on_segment_xy(
    point: tuple[float, float],
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    p = np.asarray(point, dtype=float)
    ap = p - a
    ab = b - a
    cross = abs(float(ap[0] * ab[1] - ap[1] * ab[0]))
    if cross > tolerance:
        return False
    dot = float(ap @ ab)
    if dot < -tolerance:
        return False
    return dot <= float(ab @ ab) + tolerance


def point_in_polygon_or_on_edge(point: tuple[float, float], polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    for i in range(len(polygon)):
        if point_on_segment_xy(point, polygon[i], polygon[(i + 1) % len(polygon)]):
            return True
    return point_in_polygon(point, polygon)


def box_world_xy(box: dict[str, object]) -> np.ndarray:
    corners = box.get("world_corners", box.get("box_world"))
    return np.asarray(corners, dtype=float)[:, :2]


def oriented_boxes_overlap_by_corners(a: dict[str, object], b: dict[str, object]) -> bool:
    a_xy = box_world_xy(a)
    b_xy = box_world_xy(b)
    return any(point_in_polygon_or_on_edge((float(x), float(y)), b_xy) for x, y in a_xy) or any(
        point_in_polygon_or_on_edge((float(x), float(y)), a_xy) for x, y in b_xy
    )


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


def parse_xy(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected x,y, got {value!r}")
    return float(parts[0]), float(parts[1])


def parse_bounds(value: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected min_x,max_x,min_y,max_y, got {value!r}")
    min_x, max_x, min_y, max_y = [float(p) for p in parts]
    if min_x >= max_x or min_y >= max_y:
        raise ValueError(f"Invalid bounds: {value!r}")
    return min_x, max_x, min_y, max_y


def median_or_nan(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def percentile_bounds(points: np.ndarray, pad: float = 0.03) -> tuple[float, float, float, float]:
    if points.size == 0:
        return -0.5, 0.5, -0.5, 0.5
    lo = np.percentile(points, 2, axis=0)
    hi = np.percentile(points, 98, axis=0)
    return float(lo[0] - pad), float(hi[0] + pad), float(lo[1] - pad), float(hi[1] + pad)


def sample_xy_in_hull(
    rng: random.Random,
    points_xy: np.ndarray,
    hull: np.ndarray,
    fallback_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = fallback_bounds
    if len(hull) >= 3:
        min_x, min_y = hull.min(axis=0)
        max_x, max_y = hull.max(axis=0)
        for _ in range(500):
            x = rng.uniform(float(min_x), float(max_x))
            y = rng.uniform(float(min_y), float(max_y))
            if point_in_polygon((x, y), hull):
                return x, y
    if len(points_xy):
        base = points_xy[rng.randrange(len(points_xy))]
        spread = np.std(points_xy, axis=0)
        jitter_x = max(float(spread[0]) * 0.22, 0.015)
        jitter_y = max(float(spread[1]) * 0.22, 0.015)
        return float(base[0] + rng.uniform(-jitter_x, jitter_x)), float(base[1] + rng.uniform(-jitter_y, jitter_y))
    return rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)


def aabb_overlap(
    a: tuple[float, float],
    b: tuple[float, float],
    length_m: float,
    width_m: float,
) -> bool:
    return abs(a[0] - b[0]) < length_m and abs(a[1] - b[1]) < width_m


def point_segment_distance_xy(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    ax, ay = a
    bx, by = b
    ab = np.asarray([bx - ax, by - ay], dtype=float)
    ap = np.asarray([px - ax, py - ay], dtype=float)
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(ap)), 0.0
    t = float(np.clip((ap @ ab) / denom, 0.0, 1.0))
    closest = np.asarray([ax, ay], dtype=float) + t * ab
    return float(np.linalg.norm(np.asarray([px, py], dtype=float) - closest)), t


def choose_orientation(
    rng: random.Random,
    points: list[dict[str, object]],
    keypose: str,
    task: str,
    xy: tuple[float, float],
    neighbors: int,
) -> dict[str, object] | None:
    candidates = [
        p
        for p in points
        if p.get("keypose") == keypose and (keypose == "grasp" or p.get("task") == task)
    ]
    if not candidates:
        return None
    dists = [
        math.hypot(float(p["world_x"]) - xy[0], float(p["world_y"]) - xy[1])
        for p in candidates
    ]
    order = np.argsort(np.asarray(dists, dtype=float))[: max(1, min(neighbors, len(candidates)))]
    idx = int(rng.choice([int(i) for i in order]))
    return candidates[idx]


def project_world_point(point_world: np.ndarray, model: ProjectionModel) -> tuple[float | None, float | None]:
    uv, depth = project_points(point_world.reshape(1, 3), model)
    if depth[0] <= 0:
        return None, None
    u, v = float(uv[0, 0]), float(uv[0, 1])
    if not (0 <= u <= model.width and 0 <= v <= model.height):
        return None, None
    return u, v


def project_oriented_box(
    center_world: np.ndarray,
    direction_xy: tuple[float | None, float | None],
    length_m: float,
    width_m: float,
    model: ProjectionModel,
) -> dict[str, object] | None:
    dx, dy = direction_xy
    if dx is None or dy is None:
        dx, dy = 1.0, 0.0
    axis = np.asarray([dx, dy], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        axis = np.asarray([1.0, 0.0], dtype=float)
        norm = 1.0
    axis /= norm
    perp = np.asarray([-axis[1], axis[0]], dtype=float)
    half_l = length_m / 2.0
    half_w = width_m / 2.0
    offsets_xy = [
        axis * half_l + perp * half_w,
        axis * half_l - perp * half_w,
        -axis * half_l - perp * half_w,
        -axis * half_l + perp * half_w,
    ]
    corners = np.asarray(
        [[center_world[0] + offset[0], center_world[1] + offset[1], center_world[2]] for offset in offsets_xy],
        dtype=float,
    )
    uv, depth = project_points(corners, model)
    if np.any(depth <= 0):
        return None
    if (
        np.any(uv[:, 0] < 0)
        or np.any(uv[:, 0] > model.width)
        or np.any(uv[:, 1] < 0)
        or np.any(uv[:, 1] > model.height)
    ):
        return None
    return {
        "pixel_corners": [[float(u), float(v)] for u, v in uv],
        "world_corners": [[float(x), float(y), float(z)] for x, y, z in corners],
        "direction_xy": [float(axis[0]), float(axis[1])],
    }


def box_payload_fields(box: dict[str, object] | None) -> dict[str, object]:
    if box is None:
        return {"box": None, "box_px": None, "box_world": None}
    return {
        "box": box["pixel_corners"],
        "box_px": box["pixel_corners"],
        "box_world": box["world_corners"],
        "box_direction_xy": box["direction_xy"],
    }


def random_quat_xyzw(rng: random.Random) -> np.ndarray:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    return np.asarray(
        [
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ],
        dtype=float,
    )


class IKReachability:
    def __init__(
        self,
        urdf: Path,
        right_arm_urdf: Path,
        alignment: Path,
        position_tolerance: float,
        rotation_tolerance_deg: float,
    ) -> None:
        data = json.loads(alignment.read_text(encoding="utf-8"))
        transform = data["transform_data_from_urdf"]
        self.r_data_from_urdf = np.asarray(transform["rotation_matrix"], dtype=float)
        self.t_data_from_urdf = np.asarray(transform["translation"], dtype=float)
        self.q_data_from_urdf = Rotation.from_quat(transform["rotation_quat_xyzw"])
        self.position_tolerance = position_tolerance
        self.rotation_tolerance_rad = math.radians(rotation_tolerance_deg)
        self.client = pb.connect(pb.DIRECT)
        self.fixed_dataset_joint_values = {
            FIXED_DATASET_JOINT_NAMES[col]: value
            for col, value in FIXED_DATASET_JOINT_VALUES.items()
        }
        self.fixed_torso_values = {
            FIXED_DATASET_JOINT_NAMES[col]: value
            for col, value in FIXED_DATASET_JOINT_VALUES.items()
            if 3 <= col <= 6
        }
        self.base_robot = pb.loadURDF(
            str(urdf.resolve()),
            useFixedBase=True,
            flags=pb.URDF_IGNORE_VISUAL_SHAPES,
        )
        self.base_name_to_idx = {
            pb.getJointInfo(self.base_robot, i)[1].decode(): i
            for i in range(pb.getNumJoints(self.base_robot))
        }
        for name, value in self.fixed_torso_values.items():
            if name in self.base_name_to_idx:
                pb.resetJointState(self.base_robot, self.base_name_to_idx[name], value)
        base_idx = self.base_name_to_idx["astribot_arm_right_base_fixed_joint"]
        base_state = pb.getLinkState(self.base_robot, base_idx, computeForwardKinematics=True)
        self.robot = pb.loadURDF(
            str(right_arm_urdf.resolve()),
            basePosition=base_state[4],
            baseOrientation=base_state[5],
            useFixedBase=True,
            flags=pb.URDF_IGNORE_VISUAL_SHAPES,
        )
        self.name_to_idx = {
            pb.getJointInfo(self.robot, i)[1].decode(): i
            for i in range(pb.getNumJoints(self.robot))
        }
        self.tip_idx = self.name_to_idx["astribot_arm_right_tool_joint"]
        self.movable: list[int] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.ranges: list[float] = []
        self.rest: list[float] = []
        for joint_num in range(1, 8):
            joint_name = f"astribot_arm_right_joint_{joint_num}"
            i = self.name_to_idx[joint_name]
            info = pb.getJointInfo(self.robot, i)
            lo, hi = float(info[8]), float(info[9])
            self.movable.append(i)
            self.lower.append(lo)
            self.upper.append(hi)
            self.ranges.append(hi - lo)
            self.rest.append((lo + hi) / 2.0)

    def data_to_urdf_pose(self, pos_data: np.ndarray, quat_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_urdf = self.r_data_from_urdf.T @ (pos_data - self.t_data_from_urdf)
        quat_urdf = (self.q_data_from_urdf.inv() * Rotation.from_quat(quat_data)).as_quat()
        return pos_urdf, quat_urdf

    def urdf_to_data_pose(self, pos_urdf: np.ndarray, quat_urdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_data = self.r_data_from_urdf @ pos_urdf + self.t_data_from_urdf
        quat_data = (self.q_data_from_urdf * Rotation.from_quat(quat_urdf)).as_quat()
        return pos_data, quat_data

    def solve(self, pos_data: np.ndarray, quat_data: np.ndarray) -> dict[str, object] | None:
        pos_urdf, quat_urdf = self.data_to_urdf_pose(pos_data, quat_data)
        rotation_urdf = Rotation.from_quat(quat_urdf)
        tool_pos_urdf = pos_urdf - rotation_urdf.apply(RIGHT_ARM_EE_OFFSET_TOOL)
        solution = pb.calculateInverseKinematics(
            self.robot,
            self.tip_idx,
            tool_pos_urdf,
            quat_urdf,
            lowerLimits=self.lower,
            upperLimits=self.upper,
            jointRanges=self.ranges,
            restPoses=self.rest,
            maxNumIterations=500,
            residualThreshold=1e-8,
        )
        for joint_idx, value in zip(self.movable, solution):
            pb.resetJointState(self.robot, joint_idx, float(value))
        state = pb.getLinkState(self.robot, self.tip_idx, computeForwardKinematics=True)
        solved_tool_pos_urdf = np.asarray(state[4], dtype=float)
        solved_quat_urdf = np.asarray(state[5], dtype=float)
        solved_pos_urdf = solved_tool_pos_urdf + Rotation.from_quat(solved_quat_urdf).apply(
            RIGHT_ARM_EE_OFFSET_TOOL
        )
        solved_pos_data, solved_quat_data = self.urdf_to_data_pose(
            solved_pos_urdf,
            solved_quat_urdf,
        )
        pos_error = float(np.linalg.norm(solved_pos_data - pos_data))
        rot_error = float(
            (Rotation.from_quat(solved_quat_data).inv() * Rotation.from_quat(quat_data)).magnitude()
        )
        if pos_error > self.position_tolerance or rot_error > self.rotation_tolerance_rad:
            return None
        joint_solution = [float(v) for v in solution[: len(self.movable)]]
        joint_names = [
            pb.getJointInfo(self.robot, joint_idx)[1].decode()
            for joint_idx in self.movable
        ]
        joint_limit_margins = []
        min_margin_rad = float("inf")
        min_margin_fraction = float("inf")
        min_margin_joint = None
        for name, value, lower, upper in zip(joint_names, joint_solution, self.lower, self.upper):
            range_rad = upper - lower
            lower_margin = value - lower
            upper_margin = upper - value
            margin_rad = min(lower_margin, upper_margin)
            margin_fraction = margin_rad / range_rad if range_rad > 1e-12 else float("nan")
            joint_limit_margins.append(
                {
                    "joint_name": name,
                    "position_rad": value,
                    "lower_rad": lower,
                    "upper_rad": upper,
                    "lower_margin_rad": lower_margin,
                    "upper_margin_rad": upper_margin,
                    "min_margin_rad": margin_rad,
                    "min_margin_fraction": margin_fraction,
                }
            )
            if margin_rad < min_margin_rad:
                min_margin_rad = margin_rad
                min_margin_fraction = margin_fraction
                min_margin_joint = name
        return {
            "position_error_m": pos_error,
            "rotation_error_deg": math.degrees(rot_error),
            "joint_solution": joint_solution,
            "joint_names": joint_names,
            "fixed_joint_values": self.fixed_dataset_joint_values,
            "joint_limit_margins": joint_limit_margins,
            "min_joint_limit_margin_rad": min_margin_rad,
            "min_joint_limit_margin_fraction": min_margin_fraction,
            "min_joint_limit_margin_joint": min_margin_joint,
        }


class SamplingState:
    def __init__(
        self,
        input_dir: Path,
        model_path: Path,
        alignment: Path,
        urdf: Path,
        right_arm_urdf: Path,
        background: Path,
        candidate_grid: int,
        top_k: int,
        sample_count: int,
        sample_attempts: int,
        sample_cache: Path,
        precompute_count: int,
        workspace_xy: tuple[float, float, float, float],
        grasp_height: float,
        place_heights: dict[str, float],
        ik_position_tolerance: float,
        ik_rotation_tolerance_deg: float,
        collection_dirs: dict[str, Path],
        collection_position_tolerance_m: float,
        collection_rotation_tolerance_deg: float,
        collection_scan_interval_s: float,
        collection_file_stable_s: float,
        collection_read_timeout_s: float,
        orientation_neighbors: int,
        object_length_cm: float,
        object_width_cm: float,
        block_corridor_cm: float,
        min_tool_direction_dot: float,
        right_shoulder_xy: tuple[float, float],
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
        self.alignment = alignment
        self.urdf = urdf
        self.right_arm_urdf = right_arm_urdf
        self.background = background
        self.candidate_grid = candidate_grid
        self.top_k = top_k
        self.sample_count = sample_count
        self.sample_attempts = sample_attempts
        self.sample_cache = sample_cache
        self.precompute_count = max(0, int(precompute_count))
        self.workspace_xy = workspace_xy
        self.grasp_height = grasp_height
        self.place_heights = place_heights
        self.ik = IKReachability(
            urdf=urdf,
            right_arm_urdf=right_arm_urdf,
            alignment=alignment,
            position_tolerance=ik_position_tolerance,
            rotation_tolerance_deg=ik_rotation_tolerance_deg,
        )
        self.orientation_neighbors = orientation_neighbors
        self.collection_dirs = collection_dirs
        self.collection_position_tolerance_m = max(0.0, float(collection_position_tolerance_m))
        self.collection_rotation_tolerance_deg = max(0.0, float(collection_rotation_tolerance_deg))
        self.collection_scan_interval_s = max(0.1, float(collection_scan_interval_s))
        self.collection_file_stable_s = max(0.0, float(collection_file_stable_s))
        self.collection_read_timeout_s = max(0.0, float(collection_read_timeout_s))
        self.object_length_m = object_length_cm / 100.0
        self.object_width_m = object_width_cm / 100.0
        self.block_corridor_m = block_corridor_cm / 100.0
        self.min_tool_direction_dot = min_tool_direction_dot
        self.right_shoulder_xy = right_shoulder_xy
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
        self.generation_lock = threading.Lock()
        self.pool_lock = threading.RLock()
        self.pool_condition = threading.Condition(self.pool_lock)
        self.last_scan_error: str | None = None
        self.sample_cache_error: str | None = None
        self.last_refresh_at = -1e12
        self.refresh_interval_s = 10.0
        self.groups_cache: dict[str, dict[str, object]] = {}
        self.groups_cache_key: tuple[tuple[str, object], ...] | None = None
        self.sample_pools: dict[str, list[dict[str, object]]] = {task: [] for task in TASKS}
        self.reject_totals: dict[str, dict[str, int]] = {
            task: self.empty_reject_counts() for task in TASKS
        }
        self.fill_targets: dict[str, int] = {task: self.precompute_count for task in TASKS}
        self.fill_running: set[str] = set()
        self.next_pair_index: dict[str, int] = {task: 0 for task in TASKS}
        self.recommend_indices: dict[str, int] = {task: -1 for task in TASKS}
        self.collection_lock = threading.Lock()
        self.collection_known_files: dict[str, set[str]] = {task: set() for task in TASKS}
        self.collection_pending_files: dict[str, dict[str, dict[str, object]]] = {
            task: {} for task in TASKS
        }
        self.collection_events: list[dict[str, object]] = []
        self.collection_next_event_id = 1
        self.collection_error: str | None = None
        self.init_collection_dirs()
        self.cache_signature = self.build_sample_cache_signature()
        self.load_sample_cache()
        if self.precompute_count > 0:
            for task in TASKS:
                self.ensure_pool_async(task, self.precompute_count)
        self.collection_thread = threading.Thread(
            target=self.collection_monitor_worker,
            name="sampling-guidance-collection-monitor",
            daemon=True,
        )
        self.collection_thread.start()

    def empty_reject_counts(self) -> dict[str, int]:
        return {
            "overlap": 0,
            "blocked": 0,
            "box": 0,
            "projection": 0,
            "direction": 0,
            "grasp_ik": 0,
            "place_ik": 0,
        }

    def init_collection_dirs(self) -> None:
        for task in TASKS:
            directory = self.collection_dirs[task]
            directory.mkdir(parents=True, exist_ok=True)
            self.collection_known_files[task] = {
                str(path.resolve())
                for path in list_collection_files(directory)
            }

    def push_collection_event(self, event: dict[str, object]) -> None:
        with self.collection_lock:
            event["id"] = self.collection_next_event_id
            event["time"] = time.time()
            self.collection_next_event_id += 1
            self.collection_events.append(event)
            self.collection_events = self.collection_events[-100:]

    def collection_events_payload(self, after: int) -> dict[str, object]:
        with self.collection_lock:
            events = [
                dict(event)
                for event in self.collection_events
                if int(event.get("id", 0)) > after
            ]
            last_event_id = self.collection_next_event_id - 1
            error = self.collection_error
        return {
            "events": events,
            "last_event_id": last_event_id,
            "error": error,
            "collection_dirs": {
                task: str(self.collection_dirs[task])
                for task in TASKS
            },
        }

    def current_pair_reference_unlocked(self, task: str) -> tuple[int, dict[str, object]] | None:
        current = self.recommend_indices.get(task, -1)
        if 0 <= current < len(self.sample_pools[task]):
            pair = self.sample_pools[task][current]
            if not bool(pair.get("collected", False)):
                return current, dict(pair)
        pair = self.current_recommendation_unlocked(task, advance=False)
        current = self.recommend_indices.get(task, -1)
        if pair is None or current < 0:
            return None
        return current, pair

    def collection_monitor_worker(self) -> None:
        while True:
            try:
                self.scan_collection_dirs()
                with self.collection_lock:
                    self.collection_error = None
            except Exception as exc:
                with self.collection_lock:
                    self.collection_error = f"{type(exc).__name__}: {exc}"
            time.sleep(self.collection_scan_interval_s)

    def scan_collection_dirs(self) -> None:
        now = time.monotonic()
        ready: list[tuple[str, Path, dict[str, object]]] = []
        with self.collection_lock:
            for task in TASKS:
                seen_now: set[str] = set()
                for path in list_collection_files(self.collection_dirs[task]):
                    key = str(path.resolve())
                    seen_now.add(key)
                    if key in self.collection_known_files[task]:
                        continue
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue
                    signature = (stat.st_mtime_ns, stat.st_size)
                    pending = self.collection_pending_files[task].get(key)
                    if pending is None or pending.get("signature") != signature:
                        with self.pool_condition:
                            reference = self.current_pair_reference_unlocked(task)
                        if reference is None:
                            continue
                        pair_index, pair = reference
                        self.collection_pending_files[task][key] = {
                            "signature": signature,
                            "stable_since": now,
                            "first_seen": now,
                            "pair_index": pair_index,
                            "pair": pair,
                        }
                        continue
                    stable_since = float(pending.get("stable_since", now))
                    first_seen = float(pending.get("first_seen", now))
                    if (
                        now - stable_since >= self.collection_file_stable_s
                        or now - first_seen >= self.collection_read_timeout_s
                    ):
                        ready.append((task, path, dict(pending)))
                        self.collection_pending_files[task].pop(key, None)
                        self.collection_known_files[task].add(key)
                for key in list(self.collection_pending_files[task]):
                    if key not in seen_now:
                        self.collection_pending_files[task].pop(key, None)
        for task, path, pending in ready:
            self.process_collection_file(task, path, pending)

    def extract_keypose_poses(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
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
        keyposes: dict[str, np.ndarray] = {}
        frame_values: dict[str, int] = {}
        for keypose, frame in frames.items():
            if frame is None or frame < 0 or frame >= poses.shape[0]:
                continue
            keyposes[keypose] = np.asarray(poses[int(frame)], dtype=float)
            frame_values[keypose] = int(frame)
        return keyposes, frame_values

    def compare_collection_to_pair(
        self,
        actual_poses: dict[str, np.ndarray],
        pair: dict[str, object],
    ) -> tuple[bool, list[str], dict[str, object]]:
        metrics: dict[str, object] = {}
        reasons: list[str] = []
        for keypose in KEYPOSES:
            actual = actual_poses.get(keypose)
            expected_raw = pair.get(keypose)
            if actual is None:
                reasons.append(f"{keypose} 未检测到")
                metrics[keypose] = {"missing": True}
                continue
            if not isinstance(expected_raw, dict):
                reasons.append(f"{keypose} 推荐数据缺失")
                metrics[keypose] = {"missing_expected": True}
                continue
            expected_pos = np.asarray(
                [
                    expected_raw.get("tool_x"),
                    expected_raw.get("tool_y"),
                    expected_raw.get("tool_z"),
                ],
                dtype=float,
            )
            expected_quat = np.asarray(
                [
                    expected_raw.get("qx"),
                    expected_raw.get("qy"),
                    expected_raw.get("qz"),
                    expected_raw.get("qw"),
                ],
                dtype=float,
            )
            actual_pos = np.asarray(actual[:3], dtype=float)
            actual_quat = np.asarray(actual[3:7], dtype=float)
            if not (
                np.all(np.isfinite(actual_pos))
                and np.all(np.isfinite(actual_quat))
                and np.all(np.isfinite(expected_pos))
                and np.all(np.isfinite(expected_quat))
            ):
                reasons.append(f"{keypose} pose 含无效数值")
                metrics[keypose] = {"invalid": True}
                continue
            position_error_m = float(np.linalg.norm(actual_pos - expected_pos))
            try:
                rotation_error_deg = quat_error_deg(actual_quat, expected_quat)
            except ValueError:
                reasons.append(f"{keypose} 四元数无效")
                metrics[keypose] = {
                    "position_error_m": position_error_m,
                    "invalid_rotation": True,
                }
                continue
            metrics[keypose] = {
                "position_error_m": position_error_m,
                "rotation_error_deg": rotation_error_deg,
            }
            if position_error_m >= self.collection_position_tolerance_m:
                reasons.append(
                    f"{keypose} 位置误差 {position_error_m * 100:.1f}cm >= "
                    f"{self.collection_position_tolerance_m * 100:.1f}cm"
                )
            if rotation_error_deg >= self.collection_rotation_tolerance_deg:
                reasons.append(
                    f"{keypose} 旋转误差 {rotation_error_deg:.1f}deg >= "
                    f"{self.collection_rotation_tolerance_deg:.1f}deg"
                )
        return not reasons, reasons, metrics

    def process_collection_file(self, task: str, path: Path, pending: dict[str, object]) -> None:
        pair_index = int(pending.get("pair_index", -1))
        pair = pending.get("pair")
        if not isinstance(pair, dict):
            self.push_collection_event(
                {
                    "task": task,
                    "status": "rejected",
                    "file": str(path),
                    "message": "推荐 pair 缺失，文件未校验",
                    "reasons": ["推荐 pair 缺失"],
                    "deleted": False,
                    "advance": False,
                }
            )
            return
        try:
            actual_poses, frames = self.extract_keypose_poses(path)
            ok, reasons, metrics = self.compare_collection_to_pair(actual_poses, pair)
        except Exception as exc:
            ok = False
            frames = {}
            metrics = {}
            reasons = [f"文件读取或解析失败：{type(exc).__name__}: {exc}"]
        deleted = False
        delete_error = None
        if ok:
            completed = False
            with self.pool_condition:
                if 0 <= pair_index < len(self.sample_pools[task]):
                    cached_pair = self.sample_pools[task][pair_index]
                    if cached_pair.get("index") == pair.get("index"):
                        cached_pair["collected"] = True
                        self.save_sample_cache_unlocked()
                        completed = True
                if completed:
                    self.pool_condition.notify_all()
            if not completed:
                self.push_collection_event(
                    {
                        "task": task,
                        "status": "rejected",
                        "file": str(path),
                        "pair_index": pair.get("index", pair_index),
                        "message": "采集姿态有效，但推荐 pair 已变化，文件保留且未计完成",
                        "reasons": ["推荐 pair 已变化，未计入完成"],
                        "metrics": metrics,
                        "frames": frames,
                        "deleted": False,
                        "advance": False,
                    }
                )
                return
            self.push_collection_event(
                {
                    "task": task,
                    "status": "accepted",
                    "file": str(path),
                    "pair_index": pair.get("index", pair_index),
                    "message": "采集有效，已完成当前 pair",
                    "metrics": metrics,
                    "frames": frames,
                    "advance": True,
                }
            )
            return
        try:
            path.unlink()
            deleted = True
        except FileNotFoundError:
            deleted = True
        except Exception as exc:
            delete_error = f"{type(exc).__name__}: {exc}"
        message = "采集无效，文件已删除" if deleted else f"采集无效，删除失败：{delete_error}"
        self.push_collection_event(
            {
                "task": task,
                "status": "rejected",
                "file": str(path),
                "pair_index": pair.get("index", pair_index),
                "message": message,
                "reasons": reasons,
                "metrics": metrics,
                "frames": frames,
                "deleted": deleted,
                "advance": False,
            }
        )

    def cacheable_config(self) -> dict[str, object]:
        return {
            "version": 4,
            "workspace_xy": list(self.workspace_xy),
            "grasp_height": self.grasp_height,
            "place_heights": self.place_heights,
            "object_length_m": self.object_length_m,
            "object_width_m": self.object_width_m,
            "block_corridor_m": self.block_corridor_m,
            "min_tool_direction_dot": self.min_tool_direction_dot,
            "right_shoulder_xy": list(self.right_shoulder_xy),
            "model": str(self.model_path.resolve()),
            "alignment": str(self.alignment.resolve()),
            "urdf": str(self.urdf.resolve()),
            "right_arm_urdf": str(self.right_arm_urdf.resolve()),
            "ik_position_tolerance": self.ik.position_tolerance,
            "ik_rotation_tolerance_rad": self.ik.rotation_tolerance_rad,
            "direction_axis": self.direction_axis,
            "direction_mode": self.direction_mode,
            "fixed_joints": FIXED_DATASET_JOINT_VALUES,
        }

    def display_height_for(self, task: str, keypose: str, actual_height: float) -> float:
        if task == "centrifuge":
            return self.place_heights["multidrop"]
        return actual_height

    def build_sample_cache_signature(self) -> str:
        payload = json.dumps(self.cacheable_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load_sample_cache(self) -> None:
        if not self.sample_cache.exists():
            return
        try:
            data = json.loads(self.sample_cache.read_text(encoding="utf-8"))
            if data.get("signature") != self.cache_signature:
                self.sample_cache_error = "cache ignored: configuration changed"
                return
            pools = data.get("pools", {})
            reject_totals = data.get("reject_totals", {})
            for task in TASKS:
                items = pools.get(task, [])
                if isinstance(items, list):
                    self.sample_pools[task] = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        cached_item = dict(item)
                        cached_item.setdefault("collected", False)
                        self.sample_pools[task].append(cached_item)
                totals = reject_totals.get(task, {})
                if isinstance(totals, dict):
                    self.reject_totals[task].update(
                        {key: int(totals.get(key, 0)) for key in self.reject_totals[task]}
                    )
                self.next_pair_index[task] = len(self.sample_pools[task])
            rng_state = data.get("rng_state")
            if rng_state is not None:
                self.rng.setstate(rng_state_from_json(rng_state))
        except Exception as exc:
            self.sample_cache_error = f"cache load failed: {type(exc).__name__}: {exc}"

    def save_sample_cache_unlocked(self) -> None:
        value = {
            "signature": self.cache_signature,
            "config": self.cacheable_config(),
            "updated_at": time.time(),
            "rng_state": rng_state_to_json(self.rng.getstate()),
            "pools": self.sample_pools,
            "reject_totals": self.reject_totals,
        }
        try:
            atomic_write_json(self.sample_cache, value)
            self.sample_cache_error = None
        except Exception as exc:
            self.sample_cache_error = f"cache save failed: {type(exc).__name__}: {exc}"

    def ensure_pool_async(self, task: str, count: int) -> None:
        task = task if task in TASKS else "centrifuge"
        count = max(0, int(count))
        with self.pool_condition:
            self.ensure_pool_async_unlocked(task, count)

    def ensure_pool_async_unlocked(self, task: str, count: int) -> None:
        self.fill_targets[task] = max(self.fill_targets[task], count)
        if len(self.sample_pools[task]) >= self.fill_targets[task] or task in self.fill_running:
            return
        self.fill_running.add(task)
        thread = threading.Thread(
            target=self.fill_pool_worker,
            args=(task,),
            name=f"sampling-guidance-fill-{task}",
            daemon=True,
        )
        thread.start()


    def fill_pool_worker(self, task: str) -> None:
        try:
            while True:
                with self.pool_condition:
                    target = self.fill_targets[task]
                    if len(self.sample_pools[task]) >= target:
                        return
                with self.generation_lock:
                    pair, rejects = self.try_generate_pair(task)
                with self.pool_condition:
                    for key, value in rejects.items():
                        self.reject_totals[task][key] = self.reject_totals[task].get(key, 0) + int(value)
                    if pair is not None:
                        pair["index"] = self.next_pair_index[task]
                        pair["collected"] = False
                        self.next_pair_index[task] += 1
                        self.sample_pools[task].append(pair)
                        self.save_sample_cache_unlocked()
                    self.pool_condition.notify_all()
        finally:
            with self.pool_condition:
                self.fill_running.discard(task)
                self.pool_condition.notify_all()

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
                    "world_x": float(shifted[0, 0]),
                    "world_y": float(shifted[0, 1]),
                    "world_z": float(shifted[0, 2]),
                    "tool_x": float(point[0]),
                    "tool_y": float(point[1]),
                    "tool_z": float(point[2]),
                    "qx": float(pose[3]),
                    "qy": float(pose[4]),
                    "qz": float(pose[5]),
                    "qw": float(pose[6]),
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

    def height_stats(self, points: list[dict[str, object]]) -> dict[str, object]:
        stats: dict[str, object] = {}
        grasp_heights = [float(p["world_z"]) for p in points if p.get("keypose") == "grasp"]
        stats["grasp"] = {
            "height_m": median_or_nan(grasp_heights),
            "count": len(grasp_heights),
        }
        for task in TASKS:
            place_heights = [
                float(p["world_z"])
                for p in points
                if p.get("task") == task and p.get("keypose") == "place"
            ]
            stats[f"{task}:place"] = {
                "height_m": median_or_nan(place_heights),
                "count": len(place_heights),
            }
        return stats

    def sample_pose_at(
        self,
        all_points: list[dict[str, object]],
        task: str,
        keypose: str,
        xy: tuple[float, float],
        height_m: float,
    ) -> dict[str, object] | None:
        source = choose_orientation(
            self.rng,
            all_points,
            keypose,
            task,
            xy,
            self.orientation_neighbors,
        )
        if source is None or not np.isfinite(height_m):
            return None
        quat = np.asarray([source["qx"], source["qy"], source["qz"], source["qw"]], dtype=float)
        rotation = rotation_matrix_from_xyzw(quat)
        object_point = np.asarray([xy[0], xy[1], height_m], dtype=float)
        offset = np.asarray((self.model.offsets or {}).get("shared_tool_frame", [0.0, 0.0, 0.0]), dtype=float)
        tool_point = object_point - rotation @ offset
        u, v = project_world_point(object_point, self.model)
        if u is None or v is None:
            return None
        dx, dy = direction_from_pose(
            np.asarray([tool_point[0], tool_point[1], tool_point[2], *quat], dtype=float),
            self.direction_axis,
            self.direction_mode,
        )
        direction_u = None
        direction_v = None
        if dx is not None and dy is not None:
            end_uv, end_depth = project_points(
                (object_point + np.asarray([dx, dy, 0.0], dtype=float) * 0.04).reshape(1, 3),
                self.model,
            )
            if end_depth[0] > 0:
                direction_u = float(end_uv[0, 0] - u)
                direction_v = float(end_uv[0, 1] - v)
        return {
            "task": task,
            "keypose": keypose,
            "x": u,
            "y": v,
            "world_x": float(object_point[0]),
            "world_y": float(object_point[1]),
            "world_z": float(object_point[2]),
            "tool_x": float(tool_point[0]),
            "tool_y": float(tool_point[1]),
            "tool_z": float(tool_point[2]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3]),
            "direction_u": direction_u,
            "direction_v": direction_v,
            **box_payload_fields(
                project_oriented_box(
                    object_point,
                    (dx, dy),
                    self.object_length_m,
                    self.object_width_m,
                    self.model,
                )
            ),
            "source": {
                "task": source.get("task"),
                "episode_id": source.get("episode_id"),
                "frame": source.get("frame"),
            },
        }

    def sample_ik_pose_at(
        self,
        task: str,
        keypose: str,
        xy: tuple[float, float],
        height_m: float,
    ) -> tuple[dict[str, object] | None, str | None]:
        quat = random_quat_xyzw(self.rng)
        rotation = rotation_matrix_from_xyzw(quat)
        object_point = np.asarray([xy[0], xy[1], height_m], dtype=float)
        display_height = self.display_height_for(task, keypose, height_m)
        display_point = np.asarray([xy[0], xy[1], display_height], dtype=float)
        offset = np.asarray((self.model.offsets or {}).get("shared_tool_frame", [0.0, 0.0, 0.0]), dtype=float)
        tool_point = object_point - rotation @ offset
        u, v = project_world_point(display_point, self.model)
        if u is None or v is None:
            return None, "projection"
        dx, dy = direction_from_pose(
            np.asarray([tool_point[0], tool_point[1], tool_point[2], *quat], dtype=float),
            self.direction_axis,
            self.direction_mode,
        )
        direction_u = None
        direction_v = None
        if dx is not None and dy is not None:
            end_uv, end_depth = project_points(
                (display_point + np.asarray([dx, dy, 0.0], dtype=float) * 0.04).reshape(1, 3),
                self.model,
            )
            if end_depth[0] > 0:
                direction_u = float(end_uv[0, 0] - u)
                direction_v = float(end_uv[0, 1] - v)
        box = project_oriented_box(
            display_point,
            (dx, dy),
            self.object_length_m,
            self.object_width_m,
            self.model,
        )
        if box is None:
            return None, "box"
        direction_dot = None
        if dx is not None and dy is not None and self.min_tool_direction_dot > -1.0:
            ref = np.asarray([xy[0] - self.right_shoulder_xy[0], xy[1] - self.right_shoulder_xy[1]], dtype=float)
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm > 1e-9:
                direction_dot = float((np.asarray([dx, dy], dtype=float) @ (ref / ref_norm)))
                if direction_dot < self.min_tool_direction_dot:
                    return None, "direction"
        ik = self.ik.solve(tool_point, quat)
        if ik is None:
            return None, "ik"
        return {
            "task": task,
            "keypose": keypose,
            "x": u,
            "y": v,
            "world_x": float(display_point[0]),
            "world_y": float(display_point[1]),
            "world_z": float(display_point[2]),
            "actual_world_z": float(object_point[2]),
            "tool_x": float(tool_point[0]),
            "tool_y": float(tool_point[1]),
            "tool_z": float(tool_point[2]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3]),
            "direction_u": direction_u,
            "direction_v": direction_v,
            **box_payload_fields(box),
            "tool_direction_xy": [dx, dy],
            "tool_direction_dot": direction_dot,
            "ik": ik,
        }, None

    def try_generate_pair(self, task: str) -> tuple[dict[str, object] | None, dict[str, int]]:
        reject_counts = self.empty_reject_counts()
        min_x, max_x, min_y, max_y = self.workspace_xy
        place_height = self.place_heights[task]
        for _ in range(max(1, self.sample_attempts)):
            gxy = (self.rng.uniform(min_x, max_x), self.rng.uniform(min_y, max_y))
            pxy = (self.rng.uniform(min_x, max_x), self.rng.uniform(min_y, max_y))
            if aabb_overlap(gxy, pxy, self.object_length_m, self.object_width_m):
                reject_counts["overlap"] += 1
                continue
            distance_to_line, line_t = point_segment_distance_xy(pxy, self.right_shoulder_xy, gxy)
            if 0.0 < line_t < 1.0 and distance_to_line < self.block_corridor_m:
                reject_counts["blocked"] += 1
                continue
            grasp, grasp_error = self.sample_ik_pose_at(task, "grasp", gxy, self.grasp_height)
            if grasp is None:
                reject_key = (
                    "box"
                    if grasp_error == "box"
                    else "projection"
                    if grasp_error == "projection"
                    else "direction"
                    if grasp_error == "direction"
                    else "grasp_ik"
                )
                reject_counts[reject_key] += 1
                continue
            place, place_error = self.sample_ik_pose_at(task, "place", pxy, place_height)
            if grasp is None or place is None:
                reject_key = (
                    "box"
                    if place_error == "box"
                    else "projection"
                    if place_error == "projection"
                    else "direction"
                    if place_error == "direction"
                    else "place_ik"
                )
                reject_counts[reject_key] += 1
                continue
            if oriented_boxes_overlap_by_corners(grasp, place):
                reject_counts["overlap"] += 1
                continue
            return {
                "task": task,
                "grasp": grasp,
                "place": place,
                "distance_m": float(math.hypot(pxy[0] - gxy[0], pxy[1] - gxy[1])),
                "place_to_shoulder_grasp_line_m": distance_to_line,
                "reachability": "pybullet_ik",
            }, reject_counts
        return None, reject_counts

    def sample_pairs(
        self,
        task: str,
        count: int,
    ) -> dict[str, object]:
        requested = max(1, int(count))
        self.ensure_pool_async(task, requested)
        deadline = time.monotonic() + 300.0
        with self.pool_condition:
            while len(self.sample_pools[task]) < requested and task in self.fill_running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.pool_condition.wait(timeout=min(1.0, remaining))
            pairs = [dict(pair) for pair in self.sample_pools[task][:requested]]
            reject_counts = dict(self.reject_totals[task])
            pool_count = len(self.sample_pools[task])
            target = self.fill_targets[task]
            filling = task in self.fill_running
        payload = self.sample_pairs_payload(task, pairs, requested, pool_count, target, filling)
        payload["reject_counts"] = reject_counts
        return payload

    def current_recommendation_unlocked(
        self,
        task: str,
        advance: bool,
    ) -> dict[str, object] | None:
        pool = self.sample_pools[task]
        if not pool:
            self.recommend_indices[task] = -1
            return None
        current = self.recommend_indices.get(task, -1)
        if (
            not advance
            and 0 <= current < len(pool)
            and not bool(pool[current].get("collected", False))
        ):
            return dict(pool[current])
        start = current + 1 if advance and current >= 0 else 0
        for offset in range(len(pool)):
            index = (start + offset) % len(pool)
            if not bool(pool[index].get("collected", False)):
                self.recommend_indices[task] = index
                return dict(pool[index])
        self.recommend_indices[task] = -1
        return None

    def next_recommendation(self, task: str, advance: bool) -> dict[str, object]:
        task = task if task in TASKS else "centrifuge"
        self.ensure_pool_async(task, max(1, self.fill_targets[task]))
        deadline = time.monotonic() + 300.0
        with self.pool_condition:
            while not self.sample_pools[task]:
                if task not in self.fill_running:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.pool_condition.wait(timeout=min(1.0, remaining))
            pair = self.current_recommendation_unlocked(task, advance)
            collected_count = sum(
                1 for item in self.sample_pools[task]
                if bool(item.get("collected", False))
            )
            pool_count = len(self.sample_pools[task])
            filling = task in self.fill_running
            target = self.fill_targets[task]
        batch = self.sample_pairs_payload(task, [pair] if pair else [], 1, pool_count, target, filling)
        batch["collected_count"] = collected_count
        batch["uncollected_count"] = max(0, pool_count - collected_count)
        return {
            "width": self.model.width,
            "height": self.model.height,
            "task": task,
            "samples": {},
            "batch": batch,
            "groups": {},
            "points": [],
            "cache": self.cache_status_payload(),
        }

    def sample_pairs_payload(
        self,
        task: str,
        pairs: list[dict[str, object]],
        requested: int,
        pool_count: int,
        target: int,
        filling: bool,
    ) -> dict[str, object]:
        heights = {
            "grasp": {"height_m": self.grasp_height, "source": "configured"},
            "centrifuge:place": {"height_m": self.place_heights["centrifuge"], "source": "configured"},
            "multidrop:place": {"height_m": self.place_heights["multidrop"], "source": "configured"},
        }
        return {
            "requested": requested,
            "count": len(pairs),
            "pairs": pairs,
            "reject_counts": dict(self.reject_totals[task]),
            "cache": {
                "pool_count": pool_count,
                "target": target,
                "filling": filling,
                "path": str(self.sample_cache),
            },
            "constraints": {
                "object_length_m": self.object_length_m,
                "object_width_m": self.object_width_m,
                "block_corridor_m": self.block_corridor_m,
                "right_shoulder_xy": list(self.right_shoulder_xy),
                "min_tool_direction_dot": self.min_tool_direction_dot,
                "workspace_xy": list(self.workspace_xy),
                "reachability_backend": "pybullet_ik",
                "urdf": str(self.urdf),
                "right_arm_urdf": str(self.right_arm_urdf),
                "alignment": str(self.alignment),
            },
            "world_hulls": {},
            "heights": heights,
        }

    def cache_status_payload(self) -> dict[str, object]:
        with self.pool_condition:
            return {
                "files": 0,
                "points": 0,
                "pairs": {
                    task: len(self.sample_pools[task])
                    for task in TASKS
                },
                "collected": {
                    task: sum(
                        1 for pair in self.sample_pools[task]
                        if bool(pair.get("collected", False))
                    )
                    for task in TASKS
                },
                "targets": dict(self.fill_targets),
                "filling": sorted(self.fill_running),
                "path": str(self.sample_cache),
                "error": self.last_scan_error or self.sample_cache_error,
                "input_dir": str(self.input_dir),
                "model": str(self.model_path),
                "sampling_source": "pybullet_ik",
            }

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
        raw_count = query.get("count", [str(self.sample_count)])[0]
        try:
            count = max(1, min(1000, int(raw_count)))
        except ValueError:
            count = self.sample_count
        points: list[dict[str, object]] = []
        groups: dict[str, dict[str, object]] = {}
        batch = self.sample_pairs(task, count)
        return {
            "width": self.model.width,
            "height": self.model.height,
            "task": task,
            "samples": {},
            "batch": batch,
            "groups": groups,
            "points": points,
            "cache": self.cache_status_payload(),
        }

    def background_data_url(self) -> str:
        mime = mimetypes.guess_type(str(self.background))[0] or "image/png"
        data = base64.b64encode(self.background.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def config_payload(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "precompute_count": self.precompute_count,
            "sample_cache": str(self.sample_cache),
            "tasks": list(TASKS),
            "collection": {
                "dirs": {
                    task: str(self.collection_dirs[task])
                    for task in TASKS
                },
                "position_tolerance_m": self.collection_position_tolerance_m,
                "rotation_tolerance_deg": self.collection_rotation_tolerance_deg,
            },
        }


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
        if path == "/api/recommend":
            query = parse_qs(parsed.query)
            raw_task = query.get("task", ["centrifuge"])[0]
            task = raw_task if raw_task in TASKS else "centrifuge"
            advance = query.get("advance", ["0"])[0] in {"1", "true", "True", "yes"}
            self.send_json(self.state.next_recommendation(task, advance=advance))
            return
        if path == "/api/background":
            self.send_json({"data_url": self.state.background_data_url()})
            return
        if path == "/api/config":
            self.send_json(self.state.config_payload())
            return
        if path == "/api/collection-events":
            query = parse_qs(parsed.query)
            try:
                after = int(query.get("after", ["0"])[0])
            except ValueError:
                after = 0
            self.send_json(self.state.collection_events_payload(after))
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
        alignment=args.alignment,
        urdf=args.urdf,
        right_arm_urdf=args.right_arm_urdf,
        background=args.background,
        candidate_grid=args.candidate_grid,
        top_k=args.top_k,
        sample_count=args.sample_count,
        sample_attempts=args.sample_attempts,
        sample_cache=args.sample_cache,
        precompute_count=args.precompute_count,
        workspace_xy=parse_bounds(args.workspace_xy),
        grasp_height=args.grasp_height,
        place_heights={
            "centrifuge": args.centrifuge_place_height,
            "multidrop": args.multidrop_place_height,
        },
        ik_position_tolerance=args.ik_position_tolerance,
        ik_rotation_tolerance_deg=args.ik_rotation_tolerance_deg,
        collection_dirs={
            "centrifuge": args.centrifuge_collection_dir,
            "multidrop": args.multidrop_collection_dir,
        },
        collection_position_tolerance_m=args.collection_position_tolerance_m,
        collection_rotation_tolerance_deg=args.collection_rotation_tolerance_deg,
        collection_scan_interval_s=args.collection_scan_interval_s,
        collection_file_stable_s=args.collection_file_stable_s,
        collection_read_timeout_s=args.collection_read_timeout_s,
        orientation_neighbors=args.orientation_neighbors,
        object_length_cm=args.object_length_cm,
        object_width_cm=args.object_width_cm,
        block_corridor_cm=args.block_corridor_cm,
        min_tool_direction_dot=args.min_tool_direction_dot,
        right_shoulder_xy=parse_xy(args.right_shoulder_xy),
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
    print(
        "Sampling grasp/place pairs from the PyBullet IK feasible set; "
        f"cache={args.sample_cache}, precompute={args.precompute_count} per task."
    )
    print(
        "Loaded cached pairs: "
        + ", ".join(f"{task}={len(state.sample_pools[task])}" for task in TASKS)
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
