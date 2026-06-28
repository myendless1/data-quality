"""Shared helpers for bbox-based camera calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_OUTPUT_DIR = Path("outputs/calibration_bbox")
DEFAULT_KEYPOSE_CSV = Path("outputs/astribot_analysis/keypose_stats.csv")
DEFAULT_CAMERA = "head"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEFAULT_TASK_DIRS = {
    "centrifuge": Path("/media/damoxing/datasets/astribot_tasks/myendless/centrifuge"),
    "multidrop": Path("/media/damoxing/datasets/astribot_tasks/myendless/multidrop"),
}


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
    offsets: dict[str, Any]
    residual_rmse_px: float | None = None
    residual_median_px: float | None = None
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
            },
            "extrinsics": {
                "rotation_vector": self.rvec,
                "translation": self.tvec,
                "world_to_camera_convention": "X_cam = R * X_world + t",
            },
            "object_offsets": self.offsets,
            "metrics": {
                "residual_rmse_px": self.residual_rmse_px,
                "residual_median_px": self.residual_median_px,
                "sample_count": self.sample_count,
            },
        }


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def keypose_dataframe(path: Path = DEFAULT_KEYPOSE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(
        subset=[
            "grasp_x",
            "grasp_y",
            "grasp_z",
            "place_x",
            "place_y",
            "place_z",
        ]
    ).reset_index(drop=True)


def decode_hdf5_frame(path: Path, camera: str, frame: int = 0) -> Image.Image:
    with h5py.File(path, "r") as h5:
        if f"images_dict/{camera}/rgb" not in h5:
            cameras = list(h5["images_dict"].keys()) if "images_dict" in h5 else []
            candidates = [
                item
                for item in cameras
                if f"images_dict/{item}/rgb" in h5 and f"images_dict/{item}/rgb_size" in h5
            ]
            if not candidates:
                raise KeyError(f"No decodable rgb camera found in {path}")
            camera = candidates[0]

        data = h5[f"images_dict/{camera}/rgb"]
        sizes = np.asarray(h5[f"images_dict/{camera}/rgb_size"], dtype=np.int64)
        if frame < 0 or frame >= sizes.size:
            raise IndexError(f"Frame {frame} out of range for {path}")
        starts = np.concatenate(([0], np.cumsum(sizes[:-1])))
        start = int(starts[frame])
        end = start + int(sizes[frame])
        encoded = np.asarray(data[start:end], dtype=np.uint8).tobytes()
    return Image.open(BytesIO(encoded)).convert("RGB")


def resized_rgb(image: Image.Image, size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def rotation_matrix_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def task_episode_path(task: str, episode_id: int) -> Path:
    if task not in DEFAULT_TASK_DIRS:
        raise KeyError(f"Unknown task: {task}")
    return DEFAULT_TASK_DIRS[task] / f"{task}_episode_{int(episode_id)}.hdf5"


def pose_rotation_at(path: Path, frame: int, arm: str = "right", pose_group: str = "poses_dict") -> np.ndarray:
    with h5py.File(path, "r") as h5:
        pose = np.asarray(h5[f"{pose_group}/astribot_arm_{arm}"][int(frame)], dtype=float)
    return rotation_matrix_from_xyzw(pose[3:7])


def project_points(
    points_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = rodrigues(rvec)
    points_cam = (rotation @ points_world.T).T + tvec.reshape(1, 3)
    z = points_cam[:, 2]
    uv = np.empty((points_world.shape[0], 2), dtype=float)
    uv[:, 0] = fx * points_cam[:, 0] / z + cx
    uv[:, 1] = fy * points_cam[:, 1] / z + cy
    return uv, z


def bbox_center(bbox: dict[str, Any]) -> tuple[float, float] | None:
    if not bbox:
        return None
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    if w <= 0 or h <= 0:
        return None
    return x + w / 2.0, y + h / 2.0


def load_model(path: Path) -> ProjectionModel:
    data = read_json(path, {})
    intr = data["intrinsics"]
    extr = data["extrinsics"]
    offsets = data.get("object_offsets", data.get("object_offsets_world", {}))
    metrics = data.get("metrics", {})
    return ProjectionModel(
        width=int(data.get("width", IMAGE_WIDTH)),
        height=int(data.get("height", IMAGE_HEIGHT)),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        rvec=[float(v) for v in extr["rotation_vector"]],
        tvec=[float(v) for v in extr["translation"]],
        offsets=offsets,
        residual_rmse_px=metrics.get("residual_rmse_px"),
        residual_median_px=metrics.get("residual_median_px"),
        sample_count=metrics.get("sample_count"),
    )


def offset_for(model: ProjectionModel, task: str, label: str) -> np.ndarray:
    offsets = model.offsets or {}
    if "shared_tool_frame" in offsets:
        return np.asarray(offsets["shared_tool_frame"], dtype=float)
    if task in offsets and isinstance(offsets[task], dict):
        return np.asarray(offsets[task].get(label, [0.0, 0.0, 0.0]), dtype=float)
    return np.asarray(offsets.get(label, [0.0, 0.0, 0.0]), dtype=float)


def apply_model_offset(model: ProjectionModel, point_world: np.ndarray, rotation_world_tool: np.ndarray) -> np.ndarray:
    offsets = model.offsets or {}
    if "shared_tool_frame" in offsets:
        offset_tool = np.asarray(offsets["shared_tool_frame"], dtype=float)
        return point_world + rotation_world_tool @ offset_tool
    return point_world + np.zeros(3, dtype=float)
