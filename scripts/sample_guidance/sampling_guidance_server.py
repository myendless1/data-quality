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
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
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
DEFAULT_CENTRIFUGE_COLLECTION_DIR = Path(
    "/home/astribot/Desktop/data/disk/trans/hdf5_output_centrifuge"
)
DEFAULT_MULTIDROP_COLLECTION_DIR = Path(
    "/home/astribot/Desktop/data/disk/trans/hdf5_output_multidrop"
)
COLLECTION_FILE_SUFFIXES = {".hdf5", ".h5"}
DEFAULT_COLLECTION_FINAL_DESCENT_TOLERANCE_M = 0.005  # 0.5 cm horizontal drift in the final 1 cm descent
DEFAULT_COLLECTION_DESCENT_WINDOW_M = 0.01  # final 1 cm of vertical descent
DEFAULT_COLLECTION_GRIPPER_HALF_THRESHOLD = 0.5  # processed gripper value below this = closed more than half
DEFAULT_COLLECTION_GRIPPER_HALF_CLOSE_MAX = 1  # at most one close-to-half event allowed
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

# Tolerance for the human-collection place-height sanity check. The centrifuge
# and multidrop place heights are ~7.7cm apart, so 3cm cleanly separates a
# correct task from a forgotten frontend task switch without flagging normal
# descent variation.
PLACE_HEIGHT_MISMATCH_TOLERANCE_M = 0.03
TASK_LABELS_CN = {"centrifuge": "离心机", "multidrop": "分液器"}


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
    parser.add_argument(
        "--recommend-live-stream-url",
        default="http://127.0.0.1:8088/stream.mjpg",
        help="MJPEG URL used as the live background for the recommendation view.",
    )
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
        default=DEFAULT_CENTRIFUGE_COLLECTION_DIR,
    )
    parser.add_argument(
        "--multidrop-collection-dir",
        type=Path,
        default=DEFAULT_MULTIDROP_COLLECTION_DIR,
    )
    parser.add_argument("--collection-position-tolerance-m", type=float, default=0.03)
    parser.add_argument("--collection-rotation-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--collection-scan-interval-s", type=float, default=0.8)
    parser.add_argument("--collection-file-stable-s", type=float, default=1.0)
    parser.add_argument("--collection-read-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--collection-final-descent-tolerance-m",
        type=float,
        default=DEFAULT_COLLECTION_FINAL_DESCENT_TOLERANCE_M,
        help="Max horizontal displacement (m) allowed during the final descent window.",
    )
    parser.add_argument(
        "--collection-descent-window-m",
        type=float,
        default=DEFAULT_COLLECTION_DESCENT_WINDOW_M,
        help="Vertical descent window (m) inspected at the end of the place motion.",
    )
    parser.add_argument(
        "--collection-gripper-half-threshold",
        type=float,
        default=DEFAULT_COLLECTION_GRIPPER_HALF_THRESHOLD,
        help="Processed gripper value below this counts as closed more than half.",
    )
    parser.add_argument(
        "--collection-gripper-half-close-max",
        type=int,
        default=DEFAULT_COLLECTION_GRIPPER_HALF_CLOSE_MAX,
        help="Max number of close-more-than-half events allowed in one episode.",
    )
    parser.add_argument("--workspace-xy", default="0.18,0.65,-0.45,0.20")
    parser.add_argument("--grasp-height", type=float, default=DEFAULT_GRASP_HEIGHT_M)
    parser.add_argument("--centrifuge-place-height", type=float, default=DEFAULT_CENTRIFUGE_PLACE_HEIGHT_M)
    parser.add_argument("--multidrop-place-height", type=float, default=DEFAULT_MULTIDROP_PLACE_HEIGHT_M)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.01)
    parser.add_argument("--ik-rotation-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--ik-joint-limit-margin-fraction",
        type=float,
        default=0.02,
        help="Reject IK solutions where any joint is within this fraction of its "
        "limits (0 disables; 0.02 = 2%% safety margin). PyBullet IK limits are "
        "hints, not hard constraints, so solutions at/beyond limits are physically "
        "unreachable.",
    )
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
    parser.add_argument(
        "--delete-raw-on-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When a collected HDF5 is deleted as invalid, also delete the matching "
            "raw episode_abs_<N>.hdf5 in the sibling raw dir and refresh task_info.json. "
            "The raw dir is derived as <collection_dir>.parent.parent / "
            "<collection_dir>.name with 'hdf5_output_' stripped."
        ),
    )
    parser.add_argument(
        "--auto-delete-invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a newly scanned collection file fails validation (position/rotation "
            "mismatch, missing keyposes, unreachable match, or the matched sample is "
            "already occupied), delete the HDF5 immediately instead of keeping it for "
            "manual review. A review PNG is still rendered next to the file path before "
            "deletion so the UI can show why it was rejected. The matching raw "
            "episode_abs_<N>.hdf5 is also removed when --delete-raw-on-invalid is set."
        ),
    )
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


def direction_angle_error_deg(
    actual_pose: np.ndarray,
    expected_pose: np.ndarray,
    direction_axis: str,
    direction_mode: str,
) -> float | None:
    """Angle between the XY-projected tool directions of two poses (degrees).

    Mirrors the arrow drawn on the review bboxes: take the configured tool axis,
    project it onto the world XY plane for both poses, and return the absolute
    angle between the two directed arrows. Returns None when either projection
    is degenerate (tool axis nearly perpendicular to the table).
    """
    adx, ady = direction_from_pose(actual_pose, direction_axis, direction_mode)
    edx, edy = direction_from_pose(expected_pose, direction_axis, direction_mode)
    if adx is None or edx is None:
        return None
    delta = math.atan2(ady, adx) - math.atan2(edy, edx)
    # wrap to (-pi, pi]
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


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
        joint_limit_margin_fraction: float = 0.0,
    ) -> None:
        data = json.loads(alignment.read_text(encoding="utf-8"))
        transform = data["transform_data_from_urdf"]
        self.r_data_from_urdf = np.asarray(transform["rotation_matrix"], dtype=float)
        self.t_data_from_urdf = np.asarray(transform["translation"], dtype=float)
        self.q_data_from_urdf = Rotation.from_quat(transform["rotation_quat_xyzw"])
        self.position_tolerance = position_tolerance
        self.rotation_tolerance_rad = math.radians(rotation_tolerance_deg)
        self.joint_limit_margin_fraction = max(0.0, float(joint_limit_margin_fraction))
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
        # Reject solutions at/beyond joint limits — PyBullet's lowerLimits/upperLimits
        # are null-space hints, not hard constraints, so IK can return a
        # configuration the physical robot cannot realize even when the EE
        # position error is within tolerance.
        if (
            self.joint_limit_margin_fraction > 0.0
            and min_margin_fraction < self.joint_limit_margin_fraction
        ):
            return None
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
        recommend_live_stream_url: str,
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
        ik_joint_limit_margin_fraction: float,
        collection_dirs: dict[str, Path],
        collection_position_tolerance_m: float,
        collection_rotation_tolerance_deg: float,
        collection_scan_interval_s: float,
        collection_file_stable_s: float,
        collection_read_timeout_s: float,
        collection_final_descent_tolerance_m: float,
        collection_descent_window_m: float,
        collection_gripper_half_threshold: float,
        collection_gripper_half_close_max: int,
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
        delete_raw_on_invalid: bool = True,
        auto_delete_invalid: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.model_path = model_path
        self.alignment = alignment
        self.urdf = urdf
        self.right_arm_urdf = right_arm_urdf
        self.background = background
        self.recommend_live_stream_url = recommend_live_stream_url
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
            joint_limit_margin_fraction=ik_joint_limit_margin_fraction,
        )
        self.ik_joint_limit_margin_fraction = ik_joint_limit_margin_fraction
        self.orientation_neighbors = orientation_neighbors
        self.collection_dirs = collection_dirs
        self.collection_position_tolerance_m = max(0.0, float(collection_position_tolerance_m))
        self.collection_rotation_tolerance_deg = max(0.0, float(collection_rotation_tolerance_deg))
        self.collection_scan_interval_s = max(0.1, float(collection_scan_interval_s))
        self.collection_file_stable_s = max(0.0, float(collection_file_stable_s))
        self.collection_read_timeout_s = max(0.0, float(collection_read_timeout_s))
        self.collection_final_descent_tolerance_m = max(0.0, float(collection_final_descent_tolerance_m))
        self.collection_descent_window_m = max(0.001, float(collection_descent_window_m))
        self.collection_gripper_half_threshold = float(collection_gripper_half_threshold)
        self.collection_gripper_half_close_max = max(0, int(collection_gripper_half_close_max))
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
        self.delete_raw_on_invalid = delete_raw_on_invalid
        self.auto_delete_invalid = auto_delete_invalid
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
        self.cache_signature = self.build_sample_cache_signature()
        self.load_sample_cache()
        self.init_collection_dirs()
        try:
            self.reconcile_collection_with_cache()
        except Exception as exc:
            self.collection_error = f"启动校对失败：{type(exc).__name__}: {exc}"
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
            self.collection_known_files[task] = set()

    def reconcile_collection_with_cache(self) -> None:
        """Enforce one-to-one correspondence between collected cache entries and files.

        Already-collected cache entries are authoritative: a file that is bound to a
        collected entry (by ``source_file``) is confirmed without re-running the strict
        trajectory checks — those checks only apply to *newly* collected data. A file
        not bound to any collected entry is treated as new: matched to the nearest
        uncollected sample point, validated with the full check set, and kept or deleted.
        Collected entries whose file has disappeared are unmarked so the sample point
        becomes collectable again.
        """
        for task in TASKS:
            directory = self.collection_dirs[task]
            files = list_collection_files(directory)
            surviving_paths: set[str] = set()
            bound_indices: set[int] = set()
            for path in files:
                resolved = str(path.resolve())
                try:
                    analysis = self.extract_episode_analysis(path)
                except Exception as exc:
                    self._reject_for_review(
                        task, path, -1, None,
                        [f"启动校对：文件读取失败 {type(exc).__name__}: {exc}"], {}, None,
                    )
                    continue
                with self.pool_condition:
                    # 1) Authoritative binding by source_file.
                    matched_pair = None
                    for pair in self.sample_pools[task]:
                        if not bool(pair.get("collected", False)):
                            continue
                        src = pair.get("source_file")
                        if src and str(Path(src).resolve()) == resolved:
                            matched_pair = pair
                            break
                    # 2) Rebind by pose proximity (position+rotation only) if source_file
                    #    was lost. The strict trajectory checks are NOT applied here —
                    #    already-collected data is grandfathered.
                    if matched_pair is None:
                        best = None
                        best_dist = float("inf")
                        for pair in self.sample_pools[task]:
                            if not bool(pair.get("collected", False)):
                                continue
                            idx = pair.get("index")
                            if idx in bound_indices:
                                continue
                            if not self._poses_within_tolerance(analysis, pair):
                                continue
                            d = self._pair_pose_distance(analysis, pair)
                            if d is not None and d < best_dist:
                                best_dist = d
                                best = pair
                        if best is not None:
                            best["source_file"] = resolved
                            matched_pair = best
                            self.save_sample_cache_unlocked()
                    if matched_pair is not None:
                        bound_indices.add(matched_pair.get("index"))
                        surviving_paths.add(resolved)
                        continue
                # 3) Genuinely new file: full validation against nearest uncollected.
                handled = self.process_collection_file(task, path, {})
                if handled and path.exists():
                    surviving_paths.add(str(path.resolve()))
            # Unmark collected entries whose file has disappeared.
            with self.pool_condition:
                changed = False
                for pair in self.sample_pools[task]:
                    if not bool(pair.get("collected", False)):
                        continue
                    src = pair.get("source_file")
                    if src and str(Path(src).resolve()) not in surviving_paths:
                        pair["collected"] = False
                        pair["source_file"] = None
                        changed = True
                if changed:
                    self.save_sample_cache_unlocked()
            self.collection_known_files[task] = surviving_paths

    def _poses_within_tolerance(
        self, analysis: dict[str, object], pair: dict[str, object]
    ) -> bool:
        """Position+rotation only check (original tolerance), no trajectory checks."""
        keyposes: dict[str, np.ndarray] = analysis.get("keyposes", {})  # type: ignore[assignment]
        for keypose in KEYPOSES:
            actual = keyposes.get(keypose)
            expected_raw = pair.get(keypose)
            if actual is None or not isinstance(expected_raw, dict):
                return False
            actual_pos = np.asarray(actual[:3], dtype=float)
            actual_quat = np.asarray(actual[3:7], dtype=float)
            try:
                expected_pos = np.asarray(
                    [expected_raw["tool_x"], expected_raw["tool_y"], expected_raw["tool_z"]],
                    dtype=float,
                )
                expected_quat = np.asarray(
                    [expected_raw["qx"], expected_raw["qy"], expected_raw["qz"], expected_raw["qw"]],
                    dtype=float,
                )
            except (KeyError, TypeError):
                return False
            if not (np.all(np.isfinite(actual_pos)) and np.all(np.isfinite(actual_quat))
                    and np.all(np.isfinite(expected_pos)) and np.all(np.isfinite(expected_quat))):
                return False
            # Compare OBJECT positions (tool + rotation @ shared_tool_frame offset)
            # to stay consistent with the review overlay / recommendation window,
            # which draw boxes at the object position the operator aligns to.
            expected_rot = rotation_matrix_from_xyzw(expected_quat)
            actual_rot = rotation_matrix_from_xyzw(actual_quat)
            if not (np.all(np.isfinite(expected_rot)) and np.all(np.isfinite(actual_rot))):
                return False
            expected_object = apply_model_offset(self.model, expected_pos, expected_rot)
            actual_object = apply_model_offset(self.model, actual_pos, actual_rot)
            # XOY-plane (horizontal) error only, to stay consistent with the
            # review metric (Z is invisible in the top-down overlay).
            delta_xy = actual_object[:2] - expected_object[:2]
            if float(np.linalg.norm(delta_xy)) >= self.collection_position_tolerance_m:
                return False
            try:
                expected_pose = np.asarray([*expected_pos, *expected_quat], dtype=float)
                actual_pose = np.asarray(actual[:7], dtype=float)
                direction_error = direction_angle_error_deg(
                    actual_pose,
                    expected_pose,
                    self.direction_axis,
                    self.direction_mode,
                )
                if direction_error is not None:
                    rotation_error = direction_error
                else:
                    rotation_error = quat_error_deg(actual_quat, expected_quat)
                if rotation_error >= self.collection_rotation_tolerance_deg:
                    return False
            except ValueError:
                return False
        return True

    def _pair_pose_distance(
        self, analysis: dict[str, object], pair: dict[str, object]
    ) -> float | None:
        keyposes = analysis.get("keyposes", {})
        ag = keyposes.get("grasp")
        ap = keyposes.get("place")
        if ag is None or ap is None:
            return None
        ag = np.asarray(ag[:3], dtype=float)
        ap = np.asarray(ap[:3], dtype=float)
        agq = np.asarray(ag[3:7], dtype=float) if len(ag) >= 7 else None
        apq = np.asarray(ap[3:7], dtype=float) if len(ap) >= 7 else None
        g = pair.get("grasp")
        p = pair.get("place")
        if not isinstance(g, dict) or not isinstance(p, dict):
            return None
        try:
            eg = np.asarray([g["tool_x"], g["tool_y"], g["tool_z"]], dtype=float)
            ep = np.asarray([p["tool_x"], p["tool_y"], p["tool_z"]], dtype=float)
            egq = np.asarray([g["qx"], g["qy"], g["qz"], g["qw"]], dtype=float)
            epq = np.asarray([p["qx"], p["qy"], p["qz"], p["qw"]], dtype=float)
        except (KeyError, TypeError):
            return None
        if not (np.all(np.isfinite(ag)) and np.all(np.isfinite(ap))
                and np.all(np.isfinite(eg)) and np.all(np.isfinite(ep))):
            return None
        # Distance on OBJECT positions (tool + rotation @ offset), consistent
        # with the validation and display.
        ag_obj = ag
        ap_obj = ap
        if agq is not None and np.all(np.isfinite(agq)):
            ag_rot = rotation_matrix_from_xyzw(agq)
            if np.all(np.isfinite(ag_rot)):
                ag_obj = apply_model_offset(self.model, ag, ag_rot)
        if apq is not None and np.all(np.isfinite(apq)):
            ap_rot = rotation_matrix_from_xyzw(apq)
            if np.all(np.isfinite(ap_rot)):
                ap_obj = apply_model_offset(self.model, ap, ap_rot)
        eg_rot = rotation_matrix_from_xyzw(egq)
        ep_rot = rotation_matrix_from_xyzw(epq)
        if not (np.all(np.isfinite(eg_rot)) and np.all(np.isfinite(ep_rot))):
            return None
        eg_obj = apply_model_offset(self.model, eg, eg_rot)
        ep_obj = apply_model_offset(self.model, ep, ep_rot)
        return float(np.linalg.norm(ag_obj - eg_obj) + np.linalg.norm(ap_obj - ep_obj))

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
        retry_interval = max(self.collection_scan_interval_s, 2.0)
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
                        self.collection_pending_files[task][key] = {
                            "signature": signature,
                            "stable_since": now,
                            "first_seen": now,
                            "last_attempt": 0.0,
                            "pending_notified": False,
                        }
                        continue
                    stable_since = float(pending.get("stable_since", now))
                    first_seen = float(pending.get("first_seen", now))
                    last_attempt = float(pending.get("last_attempt", 0.0))
                    is_stable = now - stable_since >= self.collection_file_stable_s
                    timed_out = now - first_seen >= self.collection_read_timeout_s
                    due_for_attempt = now - last_attempt >= retry_interval
                    if (is_stable or timed_out) and due_for_attempt:
                        ready.append((task, path, dict(pending)))
                for key in list(self.collection_pending_files[task]):
                    if key not in seen_now:
                        self.collection_pending_files[task].pop(key, None)
        for task, path, pending in ready:
            handled = self.process_collection_file(task, path, pending)
            key = str(path.resolve())
            with self.collection_lock:
                if handled:
                    self.collection_pending_files[task].pop(key, None)
                    self.collection_known_files[task].add(key)
                else:
                    entry = self.collection_pending_files[task].get(key)
                    if entry is not None:
                        entry["last_attempt"] = time.monotonic()
                        entry["pending_notified"] = True

    def extract_episode_analysis(self, path: Path) -> dict[str, object]:
        """Read the full right-arm trajectory + gripper signal and detect keypose frames."""
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
        return {
            "keyposes": keyposes,
            "frame_values": frame_values,
            "poses": poses,
            "gripper": gripper,
        }

    def extract_keypose_poses(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        analysis = self.extract_episode_analysis(path)
        return analysis["keyposes"], analysis["frame_values"]  # type: ignore[return-value]

    def final_descent_horizontal_drift(
        self, poses: np.ndarray, place_frame: int | None
    ) -> float | None:
        """Max net horizontal drift (m) over the final ``descent_window_m`` of vertical drop.

        Walks backward from the place frame until the smoothed z has risen by
        ``collection_descent_window_m``; returns the xy displacement between the
        segment endpoints, or ``None`` if no descent segment could be found.
        """
        if poses is None or place_frame is None or place_frame <= 0 or place_frame >= poses.shape[0]:
            return None
        z = np.asarray(poses[: place_frame + 1, 2], dtype=float)
        if z.size < 2:
            return None
        smoothed = smooth_centered(z, self.z_smooth_window)
        end_z = float(smoothed[-1])
        target_z = end_z + self.collection_descent_window_m
        # Walk backward to find where z first reaches target_z (start of the final descent).
        start_idx = None
        for i in range(smoothed.size - 1, -1, -1):
            if np.isfinite(smoothed[i]) and smoothed[i] >= target_z:
                start_idx = i
                break
        if start_idx is None:
            # Whole descent is shorter than the window; use the available descent start.
            finite = np.flatnonzero(np.isfinite(smoothed))
            if finite.size < 2:
                return None
            start_idx = int(finite[0])
        if start_idx >= place_frame:
            return None
        start_xy = np.asarray(poses[start_idx, :2], dtype=float)
        end_xy = np.asarray(poses[place_frame, :2], dtype=float)
        if not (np.all(np.isfinite(start_xy)) and np.all(np.isfinite(end_xy))):
            return None
        return float(np.linalg.norm(end_xy - start_xy))

    def gripper_half_close_intervals(self, gripper: np.ndarray) -> list[tuple[int, int]]:
        """Contiguous intervals where the processed gripper is closed more than half."""
        if gripper is None or gripper.size == 0:
            return []
        signal = np.isfinite(gripper) & (gripper < self.collection_gripper_half_threshold)
        intervals: list[tuple[int, int]] = []
        i = 0
        n = signal.size
        while i < n:
            if signal[i]:
                j = i
                while j + 1 < n and signal[j + 1]:
                    j += 1
                intervals.append((int(i), int(j)))
                i = j + 1
            else:
                i += 1
        return intervals

    def find_nearest_uncollected_pair_unlocked(
        self, task: str, analysis: dict[str, object]
    ) -> tuple[int, dict[str, object]] | None:
        """Return the uncollected pool entry whose grasp+place tool pose is closest."""
        keyposes = analysis.get("keyposes", {})
        actual_grasp = keyposes.get("grasp")
        actual_place = keyposes.get("place")
        if actual_grasp is None or actual_place is None:
            return None
        ag = np.asarray(actual_grasp[:3], dtype=float)
        ap = np.asarray(actual_place[:3], dtype=float)
        agq = np.asarray(actual_grasp[3:7], dtype=float)
        apq = np.asarray(actual_place[3:7], dtype=float)
        if not (
            np.all(np.isfinite(ag))
            and np.all(np.isfinite(ap))
            and np.all(np.isfinite(agq))
            and np.all(np.isfinite(apq))
        ):
            return None
        # Match on OBJECT positions (tool + rotation @ shared_tool_frame offset),
        # consistent with the review overlay and the pass/fail validation. Matching
        # on raw tool_joint positions would be skewed by the rotating offset
        # projection and could pair a collection with the wrong sample.
        ag_rot = rotation_matrix_from_xyzw(agq)
        ap_rot = rotation_matrix_from_xyzw(apq)
        if not (np.all(np.isfinite(ag_rot)) and np.all(np.isfinite(ap_rot))):
            return None
        ag_obj = apply_model_offset(self.model, ag, ag_rot)
        ap_obj = apply_model_offset(self.model, ap, ap_rot)
        best_index = -1
        best_pair: dict[str, object] | None = None
        best_dist = float("inf")
        for pair in self.sample_pools[task]:
            if bool(pair.get("collected", False)):
                continue
            g = pair.get("grasp")
            p = pair.get("place")
            if not isinstance(g, dict) or not isinstance(p, dict):
                continue
            try:
                eg = np.asarray([g["tool_x"], g["tool_y"], g["tool_z"]], dtype=float)
                ep = np.asarray([p["tool_x"], p["tool_y"], p["tool_z"]], dtype=float)
                egq = np.asarray([g["qx"], g["qy"], g["qz"], g["qw"]], dtype=float)
                epq = np.asarray([p["qx"], p["qy"], p["qz"], p["qw"]], dtype=float)
            except (KeyError, TypeError):
                continue
            if not (np.all(np.isfinite(eg)) and np.all(np.isfinite(ep))
                    and np.all(np.isfinite(egq)) and np.all(np.isfinite(epq))):
                continue
            eg_rot = rotation_matrix_from_xyzw(egq)
            ep_rot = rotation_matrix_from_xyzw(epq)
            if not (np.all(np.isfinite(eg_rot)) and np.all(np.isfinite(ep_rot))):
                continue
            eg_obj = apply_model_offset(self.model, eg, eg_rot)
            ep_obj = apply_model_offset(self.model, ep, ep_rot)
            dist = float(np.linalg.norm(ag_obj - eg_obj) + np.linalg.norm(ap_obj - ep_obj))
            if dist < best_dist:
                best_dist = dist
                best_index = int(pair.get("index", -1))
                best_pair = pair
        if best_pair is None:
            return None
        return best_index, dict(best_pair)

    def compare_collection_to_pair(
        self,
        analysis: dict[str, object],
        pair: dict[str, object],
    ) -> tuple[bool, list[str], dict[str, object]]:
        actual_poses: dict[str, np.ndarray] = analysis.get("keyposes", {})  # type: ignore[assignment]
        frame_values: dict[str, int] = analysis.get("frame_values", {})  # type: ignore[assignment]
        poses: np.ndarray = analysis.get("poses")  # type: ignore[assignment]
        gripper: np.ndarray = analysis.get("gripper")  # type: ignore[assignment]
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
            # Compare OBJECT positions (tool + rotation @ shared_tool_frame offset),
            # not raw tool_joint positions. The review overlay and recommendation
            # window draw boxes at the object position, which is what the operator
            # aligns to. Comparing tool_joint positions instead would surface the
            # rotating offset projection (~12cm) as a spurious XY error that
            # doesn't match what the operator sees.
            expected_rot = rotation_matrix_from_xyzw(expected_quat)
            actual_rot = rotation_matrix_from_xyzw(actual_quat)
            if not (np.all(np.isfinite(expected_rot)) and np.all(np.isfinite(actual_rot))):
                reasons.append(f"{keypose} 旋转矩阵无效")
                metrics[keypose] = {"invalid": True}
                continue
            expected_object = apply_model_offset(self.model, expected_pos, expected_rot)
            actual_object = apply_model_offset(self.model, actual_pos, actual_rot)
            # XOY-plane (horizontal) error only: the review overlay projects
            # object positions onto the camera image, so Z discrepancies are not
            # visible to the operator. Match the displayed geometry by ignoring
            # the Z component in the pass/fail position metric.
            delta_xy = actual_object[:2] - expected_object[:2]
            position_error_m = float(np.linalg.norm(delta_xy))
            expected_pose = np.asarray([*expected_pos, *expected_quat], dtype=float)
            actual_pose = np.asarray(actual[:7], dtype=float)
            try:
                rotation_error_3d_deg = quat_error_deg(actual_quat, expected_quat)
            except ValueError:
                reasons.append(f"{keypose} 四元数无效")
                metrics[keypose] = {
                    "position_error_m": position_error_m,
                    "invalid_rotation": True,
                }
                continue
            direction_error = direction_angle_error_deg(
                actual_pose,
                expected_pose,
                self.direction_axis,
                self.direction_mode,
            )
            # Use the in-plane tool-direction angle (the arrow shown on the
            # review bboxes) as the pass/fail metric; fall back to the full 3D
            # quaternion error only when the projected direction is degenerate.
            if direction_error is not None:
                rotation_error_deg = direction_error
            else:
                rotation_error_deg = rotation_error_3d_deg
            metrics[keypose] = {
                "position_error_m": position_error_m,
                "rotation_error_deg": rotation_error_deg,
                "rotation_error_3d_deg": rotation_error_3d_deg,
                "rotation_error_direction_deg": direction_error,
            }
            if position_error_m >= self.collection_position_tolerance_m:
                reasons.append(
                    f"{keypose} 位置误差(XY) {position_error_m * 100:.1f}cm >= "
                    f"{self.collection_position_tolerance_m * 100:.1f}cm"
                )
            if rotation_error_deg >= self.collection_rotation_tolerance_deg:
                reasons.append(
                    f"{keypose} 旋转误差 {rotation_error_deg:.1f}deg >= "
                    f"{self.collection_rotation_tolerance_deg:.1f}deg"
                )
        # Final-descent horizontal drift check.
        drift = self.final_descent_horizontal_drift(poses, frame_values.get("place"))
        if drift is not None:
            metrics["final_descent_horizontal_drift_m"] = drift
            if drift > self.collection_final_descent_tolerance_m:
                reasons.append(
                    f"末端下降最后 {self.collection_descent_window_m * 100:.1f}cm 水平位移 "
                    f"{drift * 100:.2f}cm > {self.collection_final_descent_tolerance_m * 100:.2f}cm"
                )
        else:
            metrics["final_descent_horizontal_drift_m"] = None
        # Gripper single-close check (right arm cannot close > half more than once).
        intervals = self.gripper_half_close_intervals(gripper)
        metrics["gripper_half_close_intervals"] = len(intervals)
        if len(intervals) > self.collection_gripper_half_close_max:
            reasons.append(
                f"右臂关闭夹爪超过一半的次数 {len(intervals)} > "
                f"{self.collection_gripper_half_close_max}（多次关闭夹爪）"
            )
        return not reasons, reasons, metrics

    def build_actual_keypose_record(
        self, task: str, keypose: str, pose: np.ndarray
    ) -> dict[str, object] | None:
        """Project an actually-collected tool pose into the cache record format."""
        tool_point = np.asarray(pose[:3], dtype=float)
        quat = np.asarray(pose[3:7], dtype=float)
        if not (np.all(np.isfinite(tool_point)) and np.all(np.isfinite(quat))):
            return None
        rotation = rotation_matrix_from_xyzw(quat)
        if not np.all(np.isfinite(rotation)):
            return None
        object_point = apply_model_offset(self.model, tool_point, rotation)
        display_z = float(object_point[2])
        display_point = np.asarray([object_point[0], object_point[1], display_z], dtype=float)
        u, v = project_world_point(display_point, self.model)
        if u is None or v is None:
            return None
        dx, dy = direction_from_pose(
            np.asarray([*tool_point, *quat], dtype=float), self.direction_axis, self.direction_mode
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
            display_point, (dx, dy), self.object_length_m, self.object_width_m, self.model
        )
        return {
            "task": task,
            "keypose": keypose,
            "x": float(u),
            "y": float(v),
            "world_x": float(display_point[0]),
            "world_y": float(display_point[1]),
            "world_z": float(display_point[2]),
            "actual_world_z": float(tool_point[2]),
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
        }

    def _place_height_mismatch_warning(self, task: str, analysis: dict[str, object]) -> str | None:
        """Warn if the collected place Z doesn't match the configured task.

        The place keypose is detected at the bottom of the final descent, so its
        Z is the lowest point of the place motion. The configured
        ``place_heights[task]`` is the *object* height, while the collected
        keypose is the *tool* pose, so the rotating ``shared_tool_frame`` offset
        must be applied before comparing. If that object Z is far from the
        theoretical place height — and especially if it is closer to the other
        task's place height — the operator most likely forgot to switch the
        frontend centrifuge/multidrop toggle before collecting.
        """
        keyposes = analysis.get("keyposes", {})
        place_pose = keyposes.get("place")
        if place_pose is None:
            return None
        try:
            pose = np.asarray(place_pose, dtype=float)
            tool_point = pose[:3]
            quat = pose[3:7]
        except (IndexError, TypeError, ValueError, FloatingPointError):
            return None
        if tool_point.shape[0] < 3 or quat.shape[0] < 4:
            return None
        if not (np.all(np.isfinite(tool_point)) and np.all(np.isfinite(quat))):
            return None
        rotation = rotation_matrix_from_xyzw(quat)
        if not np.all(np.isfinite(rotation)):
            return None
        object_point = apply_model_offset(self.model, tool_point, rotation)
        actual_z = float(object_point[2])
        if not np.isfinite(actual_z):
            return None
        expected_z = float(self.place_heights[task])
        this_label = TASK_LABELS_CN.get(task, task)
        if abs(actual_z - expected_z) <= PLACE_HEIGHT_MISMATCH_TOLERANCE_M:
            return None
        # Find the other task to see if the operator did the wrong task.
        other_task = next((t for t in TASKS if t != task), None)
        other_z = float(self.place_heights[other_task]) if other_task else None
        other_label = TASK_LABELS_CN.get(other_task, other_task) if other_task else None
        if (
            other_z is not None
            and abs(actual_z - other_z) < abs(actual_z - expected_z)
        ):
            return (
                f"放置高度 {actual_z * 100:.2f}cm 更接近 {other_label}"
                f"（{other_z * 100:.2f}cm）而非 {this_label}（{expected_z * 100:.2f}cm），"
                f"请确认前端任务切换正确"
            )
        return (
            f"放置高度 {actual_z * 100:.2f}cm 偏离 {this_label} 理论高度 "
            f"{expected_z * 100:.2f}cm 超过 {PLACE_HEIGHT_MISMATCH_TOLERANCE_M * 100:.1f}cm"
        )

    def process_collection_file(self, task: str, path: Path, pending: dict[str, object]) -> bool:
        """Validate a new collection file. Returns True if handled (accepted/deleted), False if it should retry."""
        try:
            analysis = self.extract_episode_analysis(path)
        except Exception as exc:
            self._reject_for_review(task, path, -1, None, [f"文件读取或解析失败：{type(exc).__name__}: {exc}"], {}, None)
            return True
        keyposes = analysis.get("keyposes", {})
        frames = analysis.get("frame_values", {})
        if keyposes.get("grasp") is None or keyposes.get("place") is None:
            self._reject_for_review(
                task, path, -1, None,
                ["未检测到 grasp/place 关键帧（夹爪闭合段或下降段缺失）"],
                {}, analysis,
            )
            return True
        place_warning = self._place_height_mismatch_warning(task, analysis)
        warnings = [place_warning] if place_warning else []
        # Match to the nearest uncollected sample point in the pool.
        with self.pool_condition:
            reference = self.find_nearest_uncollected_pair_unlocked(task, analysis)
        if reference is None:
            # Pool has no uncollected candidates yet; leave the file to retry later.
            self.push_collection_event(
                {
                    "task": task,
                    "status": "pending",
                    "file": str(path),
                    "pair_index": -1,
                    "message": "暂无未采集的采样点可匹配，稍后重试",
                    "reasons": ["无可匹配采样点"],
                    "warnings": warnings,
                    "deleted": False,
                    "advance": False,
                }
            )
            return False
        pair_index, pair = reference
        try:
            ok, reasons, metrics = self.compare_collection_to_pair(analysis, pair)
        except Exception as exc:
            ok = False
            metrics = {}
            reasons = [f"校验失败：{type(exc).__name__}: {exc}"]
        if ok:
            completed = False
            with self.pool_condition:
                cached_pair = None
                for candidate in self.sample_pools[task]:
                    if candidate.get("index") == pair.get("index") and not bool(candidate.get("collected", False)):
                        cached_pair = candidate
                        break
                if cached_pair is not None:
                    actual_grasp = self.build_actual_keypose_record(task, "grasp", keyposes["grasp"])
                    actual_place = self.build_actual_keypose_record(task, "place", keyposes["place"])
                    cached_pair["collected"] = True
                    cached_pair["source_file"] = str(path.resolve())
                    if actual_grasp is not None:
                        cached_pair["grasp"] = actual_grasp
                    if actual_place is not None:
                        cached_pair["place"] = actual_place
                    self.save_sample_cache_unlocked()
                    completed = True
                if completed:
                    self.pool_condition.notify_all()
            if not completed:
                review_image = self.render_review_image(
                    task, path, analysis, pair, ["匹配采样点已被占用"], metrics, status="rejected"
                )
                if self.auto_delete_invalid:
                    delete_result = self.delete_collection_file(str(path))
                    deleted = bool(delete_result.get("ok"))
                    message = (
                        "采集姿态有效，但匹配的采样点已被占用，已自动删除文件"
                        if deleted
                        else "采集姿态有效，但匹配的采样点已被占用，自动删除失败，文件保留且未计完成"
                    )
                    self.push_collection_event(
                        {
                            "task": task,
                            "status": "rejected",
                            "file": str(path),
                            "pair_index": pair.get("index", pair_index),
                            "message": message,
                            "reasons": ["匹配采样点已被占用"],
                            "warnings": warnings,
                            "metrics": metrics,
                            "frames": frames,
                            "deleted": deleted,
                            "advance": False,
                            "pending_review": not deleted,
                            "overlay": self._build_review_overlay(task, analysis, pair),
                            "review_image": review_image,
                        }
                    )
                    return True
                self.push_collection_event(
                    {
                        "task": task,
                        "status": "rejected",
                        "file": str(path),
                        "pair_index": pair.get("index", pair_index),
                        "message": "采集姿态有效，但匹配的采样点已被占用，文件保留且未计完成",
                        "reasons": ["匹配采样点已被占用"],
                        "warnings": warnings,
                        "metrics": metrics,
                        "frames": frames,
                        "deleted": False,
                        "advance": False,
                        "pending_review": True,
                        "overlay": self._build_review_overlay(task, analysis, pair),
                        "review_image": review_image,
                    }
                )
                return True
            review_image = self.render_review_image(
                task, path, analysis, pair, [], metrics, status="accepted"
            )
            self.push_collection_event(
                {
                    "task": task,
                    "status": "accepted",
                    "file": str(path),
                    "pair_index": pair.get("index", pair_index),
                    "message": "采集有效，已匹配并完成最近采样点",
                    "warnings": warnings,
                    "metrics": metrics,
                    "frames": frames,
                    "advance": True,
                    "overlay": self._build_review_overlay(task, analysis, pair),
                    "review_image": review_image,
                }
            )
            self._dedup_duplicate_collections(
                task, pair.get("index", pair_index), str(path), force_keep_current=True
            )
            return True
        self._reject_for_review(task, path, pair.get("index", pair_index), pair, reasons, metrics, analysis, warnings=warnings)
        return True

    def _build_review_overlay(
        self,
        task: str,
        analysis: dict[str, object] | None,
        pair: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """Project the actually-collected grasp/place keyposes + the matched reference pair.

        Each side is returned in the same record format the UI uses to draw an
        oriented bbox (x, y, box_px, direction_u/v, keypose), so the reviewer can
        overlay them on the live camera view.
        """
        if not analysis or not isinstance(pair, dict):
            return None
        keyposes = analysis.get("keyposes", {})
        ag = keyposes.get("grasp")
        ap = keyposes.get("place")
        actual_grasp = self.build_actual_keypose_record(task, "grasp", ag) if ag is not None else None
        actual_place = self.build_actual_keypose_record(task, "place", ap) if ap is not None else None
        ref_grasp = pair.get("grasp") if isinstance(pair.get("grasp"), dict) else None
        ref_place = pair.get("place") if isinstance(pair.get("place"), dict) else None
        if actual_grasp is None and actual_place is None and ref_grasp is None and ref_place is None:
            return None
        return {
            "actual_grasp": actual_grasp,
            "actual_place": actual_place,
            "reference_grasp": ref_grasp,
            "reference_place": ref_place,
        }

    def _reject_for_review(
        self,
        task: str,
        path: Path,
        pair_index: int,
        pair: dict[str, object] | None,
        reasons: list[str],
        metrics: dict[str, object],
        analysis: dict[str, object] | None,
        warnings: list[str] | None = None,
    ) -> None:
        """Reject a collection file. When ``auto_delete_invalid`` is set the file
        is deleted immediately (a review PNG is rendered first so the UI can still
        show why it was rejected); otherwise it is kept on disk for manual review.

        The kept-on-disk branch registers the file as a known file so it is not
        reprocessed on every scan. A review overlay (actual vs reference grasp/
        place projected bboxes) is attached to the event so the UI can render the
        comparison before the user decides to delete or keep it.
        """
        frames = analysis.get("frame_values", {}) if analysis else {}
        overlay = self._build_review_overlay(task, analysis, pair)
        review_image = self.render_review_image(task, path, analysis, pair, reasons, metrics, status="rejected")
        pair_idx_resolved = pair.get("index", pair_index) if pair else pair_index
        if self.auto_delete_invalid:
            delete_result = self.delete_collection_file(str(path))
            deleted = bool(delete_result.get("ok"))
            message = (
                "采集未通过校验，已自动删除文件"
                if deleted
                else "采集未通过校验，自动删除失败，文件保留待人工复核"
            )
            self.push_collection_event(
                {
                    "task": task,
                    "status": "rejected",
                    "file": str(path),
                    "pair_index": pair_idx_resolved,
                    "message": message,
                    "reasons": reasons,
                    "warnings": list(warnings) if warnings else [],
                    "metrics": metrics,
                    "frames": frames,
                    "deleted": deleted,
                    "advance": False,
                    "pending_review": not deleted,
                    "overlay": overlay,
                    "review_image": review_image,
                }
            )
            if deleted:
                return
            # deletion failed -> fall through to the keep-for-review path
        message = "采集未通过校验，文件已保留待人工复核"
        self.push_collection_event(
            {
                "task": task,
                "status": "rejected",
                "file": str(path),
                "pair_index": pair_idx_resolved,
                "message": message,
                "reasons": reasons,
                "warnings": list(warnings) if warnings else [],
                "metrics": metrics,
                "frames": frames,
                "deleted": False,
                "advance": False,
                "pending_review": True,
                "overlay": overlay,
                "review_image": review_image,
            }
        )
        self._dedup_duplicate_collections(
            task,
            pair_idx_resolved,
            str(path),
            force_keep_current=False,
        )

    def delete_collection_file(self, file_path: str) -> dict[str, object]:
        """Delete a kept collection file after manual review.

        Only files that live inside one of the configured collection directories
        may be removed, to avoid path-traversal writes outside them. When
        ``delete_raw_on_invalid`` is set, the matching raw ``episode_abs_<N>``
        file in the sibling raw data dir is also removed and ``task_info.json``
        is refreshed so the recorded Count reflects the surviving raw files.

        Any pending-review events in the in-memory buffer that reference this
        file are marked resolved (``deleted=True, pending_review=False``) so a
        page refresh does not re-trigger the review dialog.
        """
        if not file_path:
            return {"ok": False, "error": "未提供文件路径"}
        target = Path(file_path).resolve()
        allowed_roots = [Path(self.collection_dirs[task]).resolve() for task in TASKS]
        owner_task: str | None = None
        for task in TASKS:
            root = Path(self.collection_dirs[task]).resolve()
            if target == root or root in target.parents:
                owner_task = task
                break
        if owner_task is None:
            return {"ok": False, "error": "文件不在采集目录内，已拒绝"}
        key = str(target)
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            return {"ok": False, "error": f"删除失败：{type(exc).__name__}: {exc}"}
        with self.collection_lock:
            for task in TASKS:
                self.collection_known_files[task].discard(key)
                self.collection_pending_files[task].pop(key, None)
        self._resolve_pending_review_events(str(target), deleted=True, message="已手动删除采集文件")
        raw_removed: str | None = None
        if self.delete_raw_on_invalid:
            raw_removed = self._delete_raw_and_refresh_task_info(owner_task, target)
        result: dict[str, object] = {"ok": True, "file": str(target)}
        if raw_removed:
            result["raw_file"] = raw_removed
        return result

    def keep_collection_file(self, file_path: str) -> dict[str, object]:
        """Acknowledge a pending-review file as kept (no deletion).

        Marks any pending-review events in the in-memory buffer that reference
        this file as resolved (``pending_review=False``) so a page refresh does
        not re-trigger the review dialog. The file itself is left untouched.
        """
        if not file_path:
            return {"ok": False, "error": "未提供文件路径"}
        target = Path(file_path).resolve()
        allowed_roots = [Path(self.collection_dirs[task]).resolve() for task in TASKS]
        if not any(target == root or root in target.parents for root in allowed_roots):
            return {"ok": False, "error": "文件不在采集目录内，已拒绝"}
        self._resolve_pending_review_events(str(target), deleted=False, message="已保留采集文件")
        return {"ok": True, "file": str(target)}

    def _resolve_pending_review_events(
        self, file_path: str, deleted: bool, message: str
    ) -> int:
        """Mark pending-review events referencing ``file_path`` as resolved.

        Mutates matching events in the in-memory ``collection_events`` buffer so
        that a page refresh (which replays the buffer from ``after=0``) does not
        re-trigger the review dialog. Returns the number of events resolved.
        """
        resolved = 0
        with self.collection_lock:
            for event in self.collection_events:
                if (
                    event.get("file") == file_path
                    and bool(event.get("pending_review", False))
                ):
                    event["pending_review"] = False
                    event["deleted"] = bool(deleted)
                    event["message"] = message
                    resolved += 1
        return resolved

    def _raw_data_dir_for(self, task: str) -> Path | None:
        """Sibling raw data dir for a task's collection dir.

        For ``<root>/trans/hdf5_output_<task>`` this returns ``<root>/<task>``
        (i.e. go up two levels from the collection dir and drop the
        ``hdf5_output_`` prefix from the folder name). Returns None when the
        collection dir is not laid out this way or the derived raw dir is absent.
        """
        collection_dir = Path(self.collection_dirs[task]).resolve()
        raw_name = collection_dir.name.removeprefix("hdf5_output_")
        if raw_name == collection_dir.name:
            return None
        raw_dir = collection_dir.parent.parent / raw_name
        return raw_dir if raw_dir.is_dir() else None

    @staticmethod
    def _raw_filename_for(processed_name: str) -> str | None:
        """Map ``<task>_episode_<N>.hdf5`` -> ``<task>_episode_abs_<N>.hdf5``."""
        m = re.match(r"^(.*)_episode_(\d+)\.hdf5$", processed_name)
        if not m:
            return None
        return f"{m.group(1)}_episode_abs_{m.group(2)}.hdf5"

    def _refresh_task_info(self, raw_dir: Path, collection_dir: Path) -> None:
        """Recount raw ``episode_abs`` files and update ``Count``/``Disk`` in
        ``task_info.json`` for both the raw dir and the collection-dir copy."""

        def update(path: Path) -> None:
            if not path.is_file():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return
            if not isinstance(data, dict):
                return
            count = sum(
                1
                for f in raw_dir.iterdir()
                if f.name.endswith(".hdf5") and "episode_abs" in f.name
            )
            denom = ""
            cur = str(data.get("Count", ""))
            if "/" in cur:
                denom = cur.split("/", 1)[1]
            data["Count"] = f"{count}/{denom}" if denom else str(count)
            try:
                total, used, _ = shutil.disk_usage(raw_dir)
                data["Disk"] = f"{used / 1024 ** 3:.1f}/{total / 1024 ** 3:.1f}GB"
            except Exception:
                pass
            try:
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        update(raw_dir / "task_info.json")
        update(collection_dir / "task_info.json")

    def _delete_raw_and_refresh_task_info(self, task: str, processed_path: Path) -> str | None:
        raw_dir = self._raw_data_dir_for(task)
        if raw_dir is None:
            return None
        raw_name = self._raw_filename_for(processed_path.name)
        if raw_name is None:
            return None
        raw_path = raw_dir / raw_name
        removed: str | None = None
        try:
            if raw_path.is_file():
                raw_path.unlink()
                removed = str(raw_path)
        except Exception:
            removed = None
        try:
            self._refresh_task_info(raw_dir, Path(self.collection_dirs[task]).resolve())
        except Exception:
            pass
        return removed

    def _collection_score(self, metrics: dict[str, object]) -> float:
        """Composite offset-error score (lower is better).

        Each term is normalized by its tolerance so position, rotation and
        drift contribute on a common dimensionless scale. Missing/invalid
        keypose metrics add a large penalty so partial records never outrank
        complete ones.
        """
        pos_tol = max(self.collection_position_tolerance_m, 1e-6)
        rot_tol = max(self.collection_rotation_tolerance_deg, 1e-6)
        drift_tol = max(self.collection_final_descent_tolerance_m, 1e-6)
        penalty = 1e6
        total = 0.0
        for kp in KEYPOSES:
            m = metrics.get(kp) or {}
            if not isinstance(m, dict) or any(
                m.get(flag) for flag in ("missing", "missing_expected", "invalid", "invalid_rotation")
            ):
                total += penalty
                continue
            pe = m.get("position_error_m")
            re_ = m.get("rotation_error_deg")
            if pe is None or re_ is None:
                total += penalty
                continue
            total += float(pe) / pos_tol + float(re_) / rot_tol
        drift = metrics.get("final_descent_horizontal_drift_m")
        if drift is not None:
            total += float(drift) / drift_tol
        return total

    def _dedup_duplicate_collections(
        self,
        task: str,
        pair_index: int,
        keep_file: str,
        force_keep_current: bool,
    ) -> None:
        """When several collected episodes map to the same sample pair, keep
        only the best one and delete the rest.

        Called right after a fresh collection is registered. If the fresh file
        was accepted it is now the canonical collected data for the pair, so
        every prior rejected attempt is deleted (``force_keep_current=True``).
        If it was rejected, all still-pending attempts for the pair are scored
        and only the one with the smallest composite offset error survives.
        No-op when ``pair_index`` is invalid or there is only one candidate.
        """
        if pair_index is None or pair_index < 0:
            return
        candidates = [
            ev for ev in self.collection_events
            if ev.get("task") == task
            and ev.get("pair_index") == pair_index
            and not ev.get("deleted")
            and ev.get("file")
        ]
        if len(candidates) <= 1:
            return
        if force_keep_current:
            winner_file = keep_file
        else:
            winner = min(candidates, key=lambda ev: self._collection_score(ev.get("metrics") or {}))
            winner_file = winner.get("file")
        for ev in candidates:
            file_path = ev.get("file")
            if file_path == winner_file:
                continue
            self.delete_collection_file(file_path)
            ev["deleted"] = True
            self.push_collection_event(
                {
                    "task": task,
                    "status": "deduped",
                    "file": file_path,
                    "pair_index": pair_index,
                    "message": (
                        f"重复采集：已保留偏移指标最优的一条 "
                        f"({Path(winner_file).name if winner_file else '-'})，删除本条"
                    ),
                    "deleted": True,
                    "advance": False,
                }
            )

    def render_review_image(
        self,
        task: str,
        path: Path,
        analysis: dict[str, object] | None,
        pair: dict[str, object] | None,
        reasons: list[str],
        metrics: dict[str, object],
        status: str = "rejected",
    ) -> str | None:
        """Render the actual vs reference grasp/place bboxes onto the background
        image and save it next to the collected file as ``<name>.review.png``.

        Uses only the collected hdf5 (for keyposes) and the calibrated background
        image; no browser data is involved. Returns the saved path or ``None``.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return None
        overlay = self._build_review_overlay(task, analysis, pair)
        if overlay is None:
            return None
        bg_path = self.background
        if not bg_path.exists():
            return None
        try:
            img = Image.open(bg_path).convert("RGB").resize((1280, 720))
        except Exception:
            return None
        draw = ImageDraw.ImageDraw(img)
        try:
            font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
            font_text = ImageFont.truetype("DejaVuSans.ttf", 18)
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except Exception:
            font_label = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_title = ImageFont.load_default()

        def to_px(box):
            return [(float(p[0]), float(p[1])) for p in (box or [])]

        def draw_box(record, color, label):
            if not record:
                return
            box = record.get("box_px") or record.get("box")
            pts = to_px(box)
            if len(pts) >= 3:
                draw.polygon(pts, outline=color, width=5)
                # direction arrow: center -> front edge midpoint
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                fx = (pts[0][0] + pts[1][0]) / 2
                fy = (pts[0][1] + pts[1][1]) / 2
                draw.line([(cx, cy), (fx, fy)], fill=color, width=4)
            x = float(record.get("x", 0))
            y = float(record.get("y", 0))
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
            if label:
                draw.text((x + 10, y - 30), label, fill=color, font=font_label)

        ref_color = {"grasp": (42, 168, 255), "place": (255, 178, 63)}
        act_color = {"grasp": (255, 77, 109), "place": (255, 138, 59)}
        for kp in ("grasp", "place"):
            draw_box(overlay.get(f"reference_{kp}"), ref_color[kp], f"参考{kp}")
            draw_box(overlay.get(f"actual_{kp}"), act_color[kp], f"实测{kp}")

        # Legend + reasons block (top-left).
        task_labels = {"centrifuge": "离心机", "multidrop": "分液器"}
        status_label = "采集成功" if status == "accepted" else "采集未通过"
        status_color = (90, 220, 120) if status == "accepted" else (255, 90, 90)
        lines = [f"task={task_labels.get(task, task)}  pair={pair.get('index') if pair else '-'}"]
        if not reasons:
            lines.append(f"• {status_label}")
        else:
            for r in reasons:
                lines.append(f"• {r}")
        if metrics:
            for kp in ("grasp", "place"):
                m = metrics.get(kp) or {}
                pe = m.get("position_error_m")
                re_ = m.get("rotation_error_deg")
                if pe is not None and re_ is not None:
                    lines.append(f"{kp}: pos(XY)={float(pe)*100:.1f}cm rot={float(re_):.1f}deg")
        y0 = 8
        # semi-transparent backing
        bbox = draw.textbbox((10, y0), "\n".join(lines), font=font_text)
        draw.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6], fill=(0, 0, 0))
        draw.text((10, y0), "\n".join(lines), fill=(255, 255, 255), font=font_text)
        # status badge (top-right)
        try:
            sb = draw.textbbox((0, 0), status_label, font=font_title)
            sw, sh = sb[2] - sb[0], sb[3] - sb[1]
            sx, sy = 1280 - sw - 24, 12
            draw.rectangle([sx - 10, sy - 6, sx + sw + 10, sy + sh + 6], fill=(0, 0, 0))
            draw.text((sx, sy), status_label, fill=status_color, font=font_title)
        except Exception:
            pass

        out_path = path.with_name(f"{path.name}.review.png")
        try:
            img.save(out_path, "PNG")
        except Exception:
            return None
        return str(out_path)

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
        return actual_height

    def _render_display_payload(
        self,
        task: str,
        keypose: str,
        xy: tuple[float, float],
        actual_height: float,
        quat: np.ndarray,
        tool_point: np.ndarray,
    ) -> tuple[dict[str, object] | None, str | None]:
        display_height = self.display_height_for(task, keypose, actual_height)
        display_point = np.asarray([xy[0], xy[1], display_height], dtype=float)
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
        return {
            "x": u,
            "y": v,
            "world_x": float(display_point[0]),
            "world_y": float(display_point[1]),
            "world_z": float(display_point[2]),
            "actual_world_z": float(actual_height),
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
        }, None

    def build_sample_cache_signature(self) -> str:
        payload = json.dumps(self.cacheable_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ik_margin_acceptable(self, cached_item: dict[str, object]) -> bool:
        """Check whether a cached pair's IK joint margins meet the threshold.

        Returns True if the margin check is disabled (threshold == 0) or both
        grasp and place have a ``min_joint_limit_margin_fraction`` at/above the
        threshold. Pairs missing the ik field (e.g. collected pairs whose
        grasp/place were overwritten with actual-pose records) are accepted.
        """
        if self.ik_joint_limit_fraction_for_filter() <= 0.0:
            return True
        for keypose in KEYPOSES:
            rec = cached_item.get(keypose)
            if not isinstance(rec, dict):
                continue
            ik = rec.get("ik")
            if not isinstance(ik, dict):
                continue
            margin = ik.get("min_joint_limit_margin_fraction")
            if margin is None:
                continue
            try:
                if float(margin) < self.ik_joint_limit_fraction_for_filter():
                    return False
            except (TypeError, ValueError):
                continue
        return True

    def ik_joint_limit_fraction_for_filter(self) -> float:
        return getattr(self, "ik_joint_limit_margin_fraction", 0.0)

    def _recompute_cached_display_fields(self, task: str, cached_item: dict[str, object]) -> None:
        """Recompute display-only projection fields using current display logic.

        Cached pairs store the IK solution / collected tool pose (tool_x/y/z,
        quat) which stay valid across display-rule changes, but the pixel/box
        fields were projected at whatever ``display_height_for`` returned at
        sample/collect time. For centrifuge these were pinned to the multidrop
        height, so the drawn bbox was at the wrong Z. This rebuilds the display
        fields for both the ``grasp`` and ``place`` sub-records from the stored
        tool pose + quat so the UI and review image render boxes at the same Z
        where IK was actually tested / the object actually was, without
        invalidating the cache or re-running IK.
        """
        for keypose in KEYPOSES:
            rec = cached_item.get(keypose)
            if not isinstance(rec, dict):
                continue
            try:
                tool_point = np.asarray(
                    [rec.get("tool_x"), rec.get("tool_y"), rec.get("tool_z")], dtype=float
                )
                quat = np.asarray(
                    [rec.get("qx", 0.0), rec.get("qy", 0.0), rec.get("qz", 0.0), rec.get("qw", 1.0)],
                    dtype=float,
                )
            except (TypeError, ValueError):
                continue
            if (
                tool_point.shape[0] < 3
                or quat.shape[0] < 4
                or not np.all(np.isfinite(tool_point))
                or not np.all(np.isfinite(quat))
            ):
                continue
            rotation = rotation_matrix_from_xyzw(quat)
            if not np.all(np.isfinite(rotation)):
                continue
            object_point = apply_model_offset(self.model, tool_point, rotation)
            if not np.all(np.isfinite(object_point)):
                continue
            xy = (float(object_point[0]), float(object_point[1]))
            actual_height = float(object_point[2])
            display, _ = self._render_display_payload(
                task, keypose, xy, actual_height, quat, tool_point
            )
            if display is not None:
                rec.update(display)

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
            dropped = 0
            for task in TASKS:
                items = pools.get(task, [])
                if isinstance(items, list):
                    self.sample_pools[task] = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        cached_item = dict(item)
                        cached_item.setdefault("collected", False)
                        # Drop uncollected pairs whose IK solution is at/beyond
                        # joint limits — these are physically unreachable even
                        # though PyBullet IK reported a valid EE pose. Collected
                        # pairs are kept regardless (the operator already
                        # reached them; their ik field may be overwritten by
                        # actual-pose data).
                        if not bool(cached_item.get("collected", False)):
                            if not self._ik_margin_acceptable(cached_item):
                                dropped += 1
                                continue
                        self._recompute_cached_display_fields(task, cached_item)
                        self.sample_pools[task].append(cached_item)
                totals = reject_totals.get(task, {})
                if isinstance(totals, dict):
                    self.reject_totals[task].update(
                        {key: int(totals.get(key, 0)) for key in self.reject_totals[task]}
                    )
                self.next_pair_index[task] = len(self.sample_pools[task])
            if dropped > 0:
                self.sample_cache_error = (
                    f"cache load: dropped {dropped} uncollected pairs at/beyond "
                    f"joint limits (margin < {self.ik_joint_limit_margin_fraction:.2%})"
                )
            rng_state = data.get("rng_state")
            if rng_state is not None:
                self.rng.setstate(rng_state_from_json(rng_state))
            # Restore the last-shown recommendation position by pair index.
            recommend_pair_indices = data.get("recommend_pair_indices", {})
            if isinstance(recommend_pair_indices, dict):
                for task in TASKS:
                    pair_idx = int(recommend_pair_indices.get(task, -1))
                    if pair_idx < 0:
                        continue
                    pool = self.sample_pools[task]
                    found_pos = -1
                    for pos, pair in enumerate(pool):
                        if int(pair.get("index", -1)) == pair_idx:
                            found_pos = pos
                            break
                    if found_pos >= 0:
                        self.recommend_indices[task] = found_pos
                    else:
                        # Pair was dropped (e.g. by IK margin filter); land on
                        # the first uncollected pair with index >= the saved
                        # one so the operator stays roughly where they left off.
                        for pos, pair in enumerate(pool):
                            if not bool(pair.get("collected", False)) and int(pair.get("index", -1)) >= pair_idx:
                                self.recommend_indices[task] = pos
                                break
        except Exception as exc:
            self.sample_cache_error = f"cache load failed: {type(exc).__name__}: {exc}"

    def save_sample_cache_unlocked(self) -> None:
        # Persist the current recommendation position by pair index (stable ID)
        # rather than pool position, so it survives pool pruning / restarts.
        recommend_pair_indices: dict[str, int] = {}
        for task in TASKS:
            pos = self.recommend_indices.get(task, -1)
            pool = self.sample_pools.get(task, [])
            if 0 <= pos < len(pool):
                recommend_pair_indices[task] = int(pool[pos].get("index", -1))
            else:
                recommend_pair_indices[task] = -1
        value = {
            "signature": self.cache_signature,
            "config": self.cacheable_config(),
            "updated_at": time.time(),
            "rng_state": rng_state_to_json(self.rng.getstate()),
            "pools": self.sample_pools,
            "reject_totals": self.reject_totals,
            "recommend_pair_indices": recommend_pair_indices,
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
        offset = np.asarray((self.model.offsets or {}).get("shared_tool_frame", [0.0, 0.0, 0.0]), dtype=float)
        tool_point = object_point - rotation @ offset
        display, error = self._render_display_payload(task, keypose, xy, height_m, quat, tool_point)
        if display is None:
            return None, error
        dx, dy = display["tool_direction_xy"]
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
            **display,
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
        # Persist the recommendation position so a page refresh or server
        # restart lands on the same sample the operator was viewing.
        if advance:
            with self.pool_lock:
                self.save_sample_cache_unlocked()
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
            count = max(1, min(self.sample_count, int(raw_count)))
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
            "recommend_live_stream_path": "/api/head-rgbd-stream.mjpg"
            if self.recommend_live_stream_url
            else None,
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
        if path == "/api/head-rgbd-stream.mjpg":
            self.proxy_recommend_live_stream()
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
        if path == "/api/collection-delete":
            query = parse_qs(parsed.query)
            file_path = query.get("file", [""])[0]
            self.send_json(self.state.delete_collection_file(file_path))
            return
        if path == "/api/collection-keep":
            query = parse_qs(parsed.query)
            file_path = query.get("file", [""])[0]
            self.send_json(self.state.keep_collection_file(file_path))
            return
        if path == "/api/review-image":
            query = parse_qs(parsed.query)
            file_path = query.get("file", [""])[0]
            self.serve_review_image(file_path)
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

    def serve_review_image(self, file_path: str) -> None:
        """Serve a saved review PNG. Accepts either the ``.review.png`` path
        directly or the collected hdf5 path (``<name>.review.png`` is derived)."""
        if not file_path:
            self.send_error(404)
            return
        target = Path(file_path).resolve()
        allowed_roots = [Path(self.state.collection_dirs[task]).resolve() for task in TASKS]
        if not any(target == root or root in target.parents for root in allowed_roots):
            self.send_error(404)
            return
        if target.name.endswith(".review.png"):
            img_path = target
        else:
            img_path = target.with_name(f"{target.name}.review.png")
        self.serve_file(img_path)

    def proxy_recommend_live_stream(self) -> None:
        url = self.state.recommend_live_stream_url
        if not url:
            self.send_error(404)
            return
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "sampling-guidance-live-background/1.0",
                "Cache-Control": "no-cache",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as upstream:
                content_type = upstream.headers.get(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                while True:
                    chunk = upstream.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not self.wfile.closed:
                self.send_error(502, f"Live stream unavailable: {exc}")


def main() -> None:
    args = build_argparser().parse_args()
    state = SamplingState(
        input_dir=args.input_dir,
        model_path=args.model,
        alignment=args.alignment,
        urdf=args.urdf,
        right_arm_urdf=args.right_arm_urdf,
        background=args.background,
        recommend_live_stream_url=args.recommend_live_stream_url,
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
        ik_joint_limit_margin_fraction=args.ik_joint_limit_margin_fraction,
        collection_dirs={
            "centrifuge": args.centrifuge_collection_dir,
            "multidrop": args.multidrop_collection_dir,
        },
        collection_position_tolerance_m=args.collection_position_tolerance_m,
        collection_rotation_tolerance_deg=args.collection_rotation_tolerance_deg,
        collection_scan_interval_s=args.collection_scan_interval_s,
        collection_file_stable_s=args.collection_file_stable_s,
        collection_read_timeout_s=args.collection_read_timeout_s,
        collection_final_descent_tolerance_m=args.collection_final_descent_tolerance_m,
        collection_descent_window_m=args.collection_descent_window_m,
        collection_gripper_half_threshold=args.collection_gripper_half_threshold,
        collection_gripper_half_close_max=args.collection_gripper_half_close_max,
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
        delete_raw_on_invalid=args.delete_raw_on_invalid,
        auto_delete_invalid=args.auto_delete_invalid,
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
