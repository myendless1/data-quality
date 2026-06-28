#!/usr/bin/env python3
"""Rebuild the sampling-guidance pair cache from already-collected HDF5 episodes.

The episodes currently present in the centrifuge/multidrop collection directories
are treated as already-collected pairs: each file's grasp/place keypose poses are
extracted, projected through the camera model, and written into the cache pool
with ``collected = True``.  The next server start will load this cache, skip the
collected entries, and continue sampling fresh candidate pairs from the next
index onward (filling the pool up to ``--sample-count``).

Usage:
    python rebuild_cache_from_collected.py
        [--centrifuge-dir DIR] [--multidrop-dir DIR]
        [--sample-cache PATH] [--sample-count N] [--backup]

Defaults match ``start_sampling_guidance_portable.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import sampling_guidance_server as s  # noqa: E402

TASKS = ("centrifuge", "multidrop")
KEYPOSES = ("grasp", "place")
DEFAULT_MODEL_PATH = SCRIPT_DIR / "calibration" / "camera_model_shared_tool.json"
DEFAULT_ALIGNMENT_PATH = SCRIPT_DIR / "calibration" / "urdf_fk_alignment_right_arm.json"
DEFAULT_URDF_PATH = (
    SCRIPT_DIR / "astribot_descriptions" / "urdf" / "astribot_s1_urdf" / "astribot_whole_body.urdf"
)
DEFAULT_RIGHT_ARM_URDF_PATH = DEFAULT_URDF_PATH.with_name("astribot_arm_right.urdf")
DEFAULT_SAMPLE_CACHE_PATH = SCRIPT_DIR / "sampling_guidance_pair_cache.json"
DEFAULT_CENTRIFUGE_DIR = Path("/home/astribot/Desktop/data/disk/trans/hdf5_output_centrifuge")
DEFAULT_MULTIDROP_DIR = Path("/home/astribot/Desktop/data/disk/trans/hdf5_output_multidrop")

# Must match SamplingState.cacheable_config() exactly.
WORKSPACE_XY = (0.18, 0.65, -0.45, 0.20)
GRASP_HEIGHT = 0.7821216622438248
PLACE_HEIGHTS = {"centrifuge": 0.8658671177760737, "multidrop": 0.7886867696617663}
OBJECT_LENGTH_M = 0.13
OBJECT_WIDTH_M = 0.08
BLOCK_CORRIDOR_M = 0.07
MIN_TOOL_DIRECTION_DOT = 0.0
RIGHT_SHOULDER_XY = (0.0, -0.22)
IK_POSITION_TOLERANCE = 0.01
IK_ROTATION_TOLERANCE_RAD = math.radians(5.0)
DIRECTION_AXIS = "-y"
DIRECTION_MODE = "column"
FIXED_DATASET_JOINT_VALUES = {
    0: -0.019, 1: -0.008, 2: -0.086, 3: 0.594, 4: -1.192, 5: 0.597, 6: 0.002,
    23: -0.009, 24: 0.854,
}

ARM = "right"
POSE_GROUP = "poses_dict"
GRIPPER_GROUP = "poses_dict"
LOW_GRIPPER_THRESHOLD = 0.11
Z_SMOOTH_WINDOW = 15
INVERT_GRIPPER_VALUE = True


def cacheable_config() -> dict[str, object]:
    return {
        "version": 4,
        "workspace_xy": list(WORKSPACE_XY),
        "grasp_height": GRASP_HEIGHT,
        "place_heights": PLACE_HEIGHTS,
        "object_length_m": OBJECT_LENGTH_M,
        "object_width_m": OBJECT_WIDTH_M,
        "block_corridor_m": BLOCK_CORRIDOR_M,
        "min_tool_direction_dot": MIN_TOOL_DIRECTION_DOT,
        "right_shoulder_xy": list(RIGHT_SHOULDER_XY),
        "model": str(DEFAULT_MODEL_PATH.resolve()),
        "alignment": str(DEFAULT_ALIGNMENT_PATH.resolve()),
        "urdf": str(DEFAULT_URDF_PATH.resolve()),
        "right_arm_urdf": str(DEFAULT_RIGHT_ARM_URDF_PATH.resolve()),
        "ik_position_tolerance": IK_POSITION_TOLERANCE,
        "ik_rotation_tolerance_rad": IK_ROTATION_TOLERANCE_RAD,
        "direction_axis": DIRECTION_AXIS,
        "direction_mode": DIRECTION_MODE,
        "fixed_joints": FIXED_DATASET_JOINT_VALUES,
    }


def cache_signature() -> str:
    payload = json.dumps(cacheable_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_keypose_poses(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        raw_gripper = np.asarray(h5[f"{GRIPPER_GROUP}/astribot_gripper_{ARM}"])
        poses = np.asarray(h5[f"{POSE_GROUP}/astribot_arm_{ARM}"], dtype=float)
    gripper = s.prepare_gripper_values(raw_gripper, INVERT_GRIPPER_VALUE)
    frames = s.detect_keypose_frames(
        gripper, poses[:, 2], LOW_GRIPPER_THRESHOLD, Z_SMOOTH_WINDOW
    )
    out: dict[str, np.ndarray] = {}
    for keypose, frame in frames.items():
        if frame is None or frame < 0 or frame >= poses.shape[0]:
            continue
        out[keypose] = np.asarray(poses[int(frame)], dtype=float)
    return out


def build_keypose_record(task: str, keypose: str, pose: np.ndarray, model) -> dict[str, object] | None:
    tool_point = np.asarray(pose[:3], dtype=float)
    quat = np.asarray(pose[3:7], dtype=float)
    if not (np.all(np.isfinite(tool_point)) and np.all(np.isfinite(quat))):
        return None
    rotation = s.rotation_matrix_from_xyzw(quat)
    if not np.all(np.isfinite(rotation)):
        return None
    object_point = s.apply_model_offset(model, tool_point, rotation)
    # display_height_for: centrifuge uses multidrop place height; others use actual.
    display_z = PLACE_HEIGHTS["multidrop"] if task == "centrifuge" else float(object_point[2])
    display_point = np.asarray([object_point[0], object_point[1], display_z], dtype=float)
    u, v = s.project_world_point(display_point, model)
    if u is None or v is None:
        return None
    dx, dy = s.direction_from_pose(
        np.asarray([*tool_point, *quat], dtype=float), DIRECTION_AXIS, DIRECTION_MODE
    )
    direction_u = None
    direction_v = None
    if dx is not None and dy is not None:
        end_uv, end_depth = s.project_points(
            (display_point + np.asarray([dx, dy, 0.0], dtype=float) * 0.04).reshape(1, 3),
            model,
        )
        if end_depth[0] > 0:
            direction_u = float(end_uv[0, 0] - u)
            direction_v = float(end_uv[0, 1] - v)
    box = s.project_oriented_box(
        display_point, (dx, dy), OBJECT_LENGTH_M, OBJECT_WIDTH_M, model
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
        **s.box_payload_fields(box),
        "tool_direction_xy": [dx, dy],
    }


def build_pair(task: str, index: int, path: Path, model) -> dict[str, object] | None:
    poses = extract_keypose_poses(path)
    grasp_pose = poses.get("grasp")
    place_pose = poses.get("place")
    if grasp_pose is None or place_pose is None:
        return None
    grasp = build_keypose_record(task, "grasp", grasp_pose, model)
    place = build_keypose_record(task, "place", place_pose, model)
    if grasp is None or place is None:
        return None
    gxy = (grasp["world_x"], grasp["world_y"])
    pxy = (place["world_x"], place["world_y"])
    distance_m = float(math.hypot(pxy[0] - gxy[0], pxy[1] - gxy[1]))
    dist_to_line, _ = s.point_segment_distance_xy(pxy, RIGHT_SHOULDER_XY, gxy)
    return {
        "task": task,
        "grasp": grasp,
        "place": place,
        "distance_m": distance_m,
        "place_to_shoulder_grasp_line_m": float(dist_to_line),
        "reachability": "collected",
        "index": index,
        "collected": True,
        "source_file": str(path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centrifuge-dir", type=Path, default=DEFAULT_CENTRIFUGE_DIR)
    parser.add_argument("--multidrop-dir", type=Path, default=DEFAULT_MULTIDROP_DIR)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE_PATH)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Back up the existing cache before overwriting.")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    args = parser.parse_args()

    model = s.load_model(DEFAULT_MODEL_PATH)
    collection_dirs = {"centrifuge": args.centrifuge_dir, "multidrop": args.multidrop_dir}

    pools: dict[str, list[dict[str, object]]] = {task: [] for task in TASKS}
    reject_totals = {
        task: {
            "overlap": 0, "blocked": 0, "box": 0, "projection": 0,
            "direction": 0, "grasp_ik": 0, "place_ik": 0,
        }
        for task in TASKS
    }

    for task in TASKS:
        directory = collection_dirs[task]
        files = s.list_collection_files(directory)
        index = 0
        skipped: list[str] = []
        for path in files:
            pair = build_pair(task, index, path, model)
            if pair is None:
                skipped.append(path.name)
                continue
            pools[task].append(pair)
            index += 1
        print(f"{task}: {len(pools[task])} collected pairs from {directory}")
        if skipped:
            print(f"  skipped {len(skipped)} file(s) missing grasp/place poses:")
            for name in skipped[:10]:
                print(f"    - {name}")
            if len(skipped) > 10:
                print(f"    ... ({len(skipped) - 10} more)")

    cache_path = args.sample_cache
    if args.backup and cache_path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = cache_path.with_name(f"{cache_path.name}.bak.{stamp}")
        shutil.copy2(cache_path, backup)
        print(f"backed up existing cache to {backup}")

    value = {
        "signature": cache_signature(),
        "config": cacheable_config(),
        "updated_at": time.time(),
        "rng_state": s.rng_state_to_json(random.Random().getstate()),
        "pools": pools,
        "reject_totals": reject_totals,
    }
    s.atomic_write_json(cache_path, value)
    print(f"wrote cache: {cache_path}")
    print(
        "collected: "
        + ", ".join(f"{task}={sum(1 for p in pools[task] if p.get('collected'))}" for task in TASKS)
    )
    print(
        "On next server start, sampling will continue from index "
        + ", ".join(f"{task}={len(pools[task])}" for task in TASKS)
        + f" and fill up to sample_count={args.sample_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
