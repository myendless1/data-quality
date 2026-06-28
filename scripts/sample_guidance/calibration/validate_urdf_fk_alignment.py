#!/usr/bin/env python3
"""Validate Astribot URDF FK/IK alignment against recorded HDF5 poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_GUIDANCE_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = Path("/media/damoxing/datasets/astribot_tasks/myendless")
DEFAULT_URDF = (
    SAMPLE_GUIDANCE_DIR
    / "astribot_descriptions"
    / "urdf"
    / "astribot_s1_urdf"
    / "astribot_whole_body.urdf"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "urdf_fk_alignment_right_arm.json"
TASKS = ("centrifuge", "multidrop")

JOINT_COLUMN_MAP = {
    "torso": list(range(3, 7)),
    "left_arm": list(range(7, 14)),
    "right_arm": list(range(15, 22)),
    "right_gripper": [22],
    "head": [23, 24],
}


def episode_id(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--files-per-task", type=int, default=4)
    parser.add_argument("--frames-per-file", type=int, default=60)
    parser.add_argument("--ik-frames-per-task", type=int, default=12)
    return parser


def load_robot(urdf: Path) -> tuple[int, dict[str, int], list[int]]:
    robot = pb.loadURDF(str(urdf.resolve()), useFixedBase=True, flags=pb.URDF_IGNORE_VISUAL_SHAPES)
    name_to_idx = {pb.getJointInfo(robot, i)[1].decode(): i for i in range(pb.getNumJoints(robot))}
    movable = [
        i
        for i in range(pb.getNumJoints(robot))
        if pb.getJointInfo(robot, i)[2] == pb.JOINT_REVOLUTE
    ]
    return robot, name_to_idx, movable


def set_recorded_state(
    robot: int,
    name_to_idx: dict[str, int],
    joints: np.ndarray,
    frame: int,
) -> None:
    for i in range(pb.getNumJoints(robot)):
        pb.resetJointState(robot, i, 0.0)
    for col, joint_idx in zip(JOINT_COLUMN_MAP["torso"], [name_to_idx[f"astribot_torso_joint_{i}"] for i in range(1, 5)]):
        pb.resetJointState(robot, joint_idx, float(joints[frame, col]))
    for col, joint_idx in zip(JOINT_COLUMN_MAP["left_arm"], [name_to_idx[f"astribot_arm_left_joint_{i}"] for i in range(1, 8)]):
        pb.resetJointState(robot, joint_idx, float(joints[frame, col]))
    for col, joint_idx in zip(JOINT_COLUMN_MAP["right_arm"], [name_to_idx[f"astribot_arm_right_joint_{i}"] for i in range(1, 8)]):
        pb.resetJointState(robot, joint_idx, float(joints[frame, col]))


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = build_argparser().parse_args()
    pb.connect(pb.DIRECT)
    robot, name_to_idx, movable = load_robot(args.urdf)
    tip_idx = name_to_idx["astribot_arm_right_ee_joint"]

    fk_positions: list[np.ndarray] = []
    data_positions: list[np.ndarray] = []
    fk_quats: list[np.ndarray] = []
    data_quats: list[np.ndarray] = []
    files: list[str] = []

    for task in TASKS:
        paths = sorted((args.input_dir / task).glob("*.hdf5"), key=episode_id)[: args.files_per_task]
        for path in paths:
            files.append(str(path))
            with h5py.File(path, "r") as h5:
                joints = np.asarray(h5["joints_dict/joints_position_state"], dtype=float)
                poses = np.asarray(h5["poses_dict/astribot_arm_right"], dtype=float)
            frames = np.linspace(0, len(joints) - 1, args.frames_per_file, dtype=int)
            for frame in frames:
                set_recorded_state(robot, name_to_idx, joints, int(frame))
                state = pb.getLinkState(robot, tip_idx, computeForwardKinematics=True)
                fk_positions.append(np.asarray(state[4], dtype=float))
                fk_quats.append(np.asarray(state[5], dtype=float))
                data_positions.append(poses[frame, :3])
                data_quats.append(poses[frame, 3:7])

    fk_pos = np.asarray(fk_positions, dtype=float)
    data_pos = np.asarray(data_positions, dtype=float)
    fk_rot = Rotation.from_quat(np.asarray(fk_quats, dtype=float))
    data_rot = Rotation.from_quat(np.asarray(data_quats, dtype=float))

    r_data_from_urdf, t_data_from_urdf = rigid_alignment(fk_pos, data_pos)
    pos_raw = np.linalg.norm(fk_pos - data_pos, axis=1)
    pos_aligned = np.linalg.norm((r_data_from_urdf @ fk_pos.T).T + t_data_from_urdf - data_pos, axis=1)

    delta = data_rot * fk_rot.inv()
    q_data_from_urdf = delta.mean()
    rot_residual_deg = np.degrees((q_data_from_urdf.inv() * delta).magnitude())

    lower: list[float] = []
    upper: list[float] = []
    ranges: list[float] = []
    rest: list[float] = []
    for joint_idx in movable:
        info = pb.getJointInfo(robot, joint_idx)
        lo, hi = float(info[8]), float(info[9])
        lower.append(lo)
        upper.append(hi)
        ranges.append(hi - lo)
        rest.append((lo + hi) / 2.0)

    ik_pos_errors: list[float] = []
    ik_rot_errors_deg: list[float] = []
    ik_frames_per_task = max(1, args.ik_frames_per_task)
    for task in TASKS:
        path = sorted((args.input_dir / task).glob("*.hdf5"), key=episode_id)[0]
        with h5py.File(path, "r") as h5:
            poses = np.asarray(h5["poses_dict/astribot_arm_right"], dtype=float)
        frames = np.linspace(0, len(poses) - 1, ik_frames_per_task, dtype=int)
        for frame in frames:
            data_p = poses[frame, :3]
            data_r = Rotation.from_quat(poses[frame, 3:7])
            urdf_p = r_data_from_urdf.T @ (data_p - t_data_from_urdf)
            urdf_r = q_data_from_urdf.inv() * data_r
            solution = pb.calculateInverseKinematics(
                robot,
                tip_idx,
                urdf_p,
                urdf_r.as_quat(),
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=500,
                residualThreshold=1e-8,
            )
            for joint_idx, value in zip(movable, solution):
                pb.resetJointState(robot, joint_idx, float(value))
            state = pb.getLinkState(robot, tip_idx, computeForwardKinematics=True)
            solved_data_p = r_data_from_urdf @ np.asarray(state[4], dtype=float) + t_data_from_urdf
            solved_data_r = q_data_from_urdf * Rotation.from_quat(state[5])
            ik_pos_errors.append(float(np.linalg.norm(solved_data_p - data_p)))
            ik_rot_errors_deg.append(float(np.degrees((solved_data_r.inv() * data_r).magnitude())))

    result = {
        "urdf": str(args.urdf),
        "tip_link": "astribot_arm_right_end_effector",
        "tip_joint_index": tip_idx,
        "joint_column_map": JOINT_COLUMN_MAP,
        "sampled_files": files,
        "frames": int(len(fk_pos)),
        "transform_data_from_urdf": {
            "rotation_matrix": r_data_from_urdf.tolist(),
            "translation": t_data_from_urdf.tolist(),
            "rotation_quat_xyzw": q_data_from_urdf.as_quat().tolist(),
        },
        "fk_position_error_raw_m": summarize(pos_raw),
        "fk_position_error_aligned_m": summarize(pos_aligned),
        "fk_rotation_error_aligned_deg": summarize(rot_residual_deg),
        "ik_roundtrip_position_error_m": summarize(np.asarray(ik_pos_errors, dtype=float)),
        "ik_roundtrip_rotation_error_deg": summarize(np.asarray(ik_rot_errors_deg, dtype=float)),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    pb.disconnect()


if __name__ == "__main__":
    main()
