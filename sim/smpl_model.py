from __future__ import annotations

from functools import lru_cache
import pickle
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


MALE_MODEL_FILENAME = "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"


class SmplModel:
    """Small, self-contained SMPL loader and linear-blend-skinning runtime."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_path = Path(model_dir) / MALE_MODEL_FILENAME
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SMPL male model is missing: {self.model_path}. "
                "Download the registered SMPL Python model and mount/pass its model directory."
            )
        self.model = _load_model(str(self.model_path.resolve()))

    @property
    def faces(self) -> np.ndarray:
        return self.model["faces"]

    def canonical_vertices(
        self,
        betas: list[float] | np.ndarray,
        correction: dict[str, Any] | None = None,
        segmentation: dict[str, list[int]] | None = None,
    ) -> np.ndarray:
        values = np.zeros(self.model["shapedirs"].shape[-1], dtype=np.float64)
        source = np.asarray(betas, dtype=np.float64).reshape(-1)
        values[: min(len(values), len(source))] = source[: min(len(values), len(source))]
        vertices = self.model["v_template"] + np.tensordot(self.model["shapedirs"], values, axes=([2], [0]))
        if correction:
            if not segmentation:
                raise ValueError("segmentation is required when applying body correction")
            vertices = apply_body_correction(vertices, correction, segmentation)
        return vertices

    def posed_vertices(
        self,
        params: dict[str, Any],
        correction: dict[str, Any] | None = None,
        segmentation: dict[str, list[int]] | None = None,
    ) -> np.ndarray:
        v_shaped = self.canonical_vertices(params.get("betas") or [], correction, segmentation)
        joints = np.asarray(self.model["j_regressor"].dot(v_shaped), dtype=np.float64)
        rotations = batch_rodrigues(_pose_vector(params).reshape(-1, 3))
        pose_feature = (rotations[1:] - np.eye(3, dtype=np.float64)).reshape(-1)
        v_posed = v_shaped + np.tensordot(self.model["posedirs"], pose_feature, axes=([2], [0]))
        transforms = global_rigid_transforms(rotations, joints, self.model["parents"])
        joint_h = np.concatenate([joints, np.zeros((24, 1), dtype=np.float64)], axis=1)
        transforms = transforms - pack(np.matmul(transforms, joint_h.reshape(24, 4, 1)))
        blended = np.tensordot(self.model["weights"], transforms.reshape(24, 16), axes=([1], [0])).reshape(-1, 4, 4)
        v_h = np.concatenate([v_posed, np.ones((len(v_posed), 1), dtype=np.float64)], axis=1)
        vertices = np.matmul(blended, v_h.reshape(-1, 4, 1))[:, :3, 0]
        vertices *= float(params.get("body_scale") or 1.0)
        vertices[:, 1] -= float(vertices[:, 1].min())
        translation = np.asarray(params.get("transl") or [0.0, 0.0, 0.0], dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError("body params transl must contain exactly 3 values")
        vertices += translation.reshape(1, 3)
        if not np.isfinite(vertices).all():
            raise ValueError("SMPL generation produced non-finite vertices")
        return vertices


def apply_body_correction(
    vertices: np.ndarray,
    correction: dict[str, Any],
    segmentation: dict[str, list[int]],
) -> np.ndarray:
    """Apply bounded, symmetric corrections in the canonical SMPL frame."""
    result = np.asarray(vertices, dtype=np.float64).copy()
    y_min = float(result[:, 1].min())
    height = max(float(result[:, 1].max()) - y_min, 1.0e-8)
    normalized_y = (result[:, 1] - y_min) / height
    torso = correction.get("torso_radial", {})
    centers = {"hip": 0.48, "waist": 0.55, "chest": 0.72}
    sigma = 0.055
    weights = {name: np.exp(-0.5 * ((normalized_y - center) / sigma) ** 2) for name, center in centers.items()}
    numerator = np.zeros(len(result), dtype=np.float64)
    denominator = np.zeros(len(result), dtype=np.float64)
    for name, weight in weights.items():
        numerator += weight * (float(torso.get(name, 1.0)) - 1.0)
        denominator += weight
    radial = 1.0 + np.divide(numerator, np.maximum(denominator, 1.0), out=np.zeros_like(numerator), where=True)
    result[:, 0] *= radial
    result[:, 2] *= radial

    shoulder_factor = float(correction.get("shoulder_width", 1.0))
    shoulder_y = _segment_y(result, segmentation, ("leftShoulder", "rightShoulder"), default=0.80)
    shoulder_weight = np.exp(-0.5 * ((normalized_y - shoulder_y) / 0.055) ** 2)
    result[:, 0] *= 1.0 + (shoulder_factor - 1.0) * shoulder_weight

    arm_factor = float(correction.get("arm_length", 1.0))
    if abs(arm_factor - 1.0) > 1.0e-12:
        for side in ("left", "right"):
            shoulder_ids = _valid_indices(segmentation.get(f"{side}Shoulder", []), len(result))
            arm_ids = _valid_indices(
                segmentation.get(f"{side}Arm", [])
                + segmentation.get(f"{side}ForeArm", [])
                + segmentation.get(f"{side}Hand", []),
                len(result),
            )
            if not shoulder_ids or not arm_ids:
                continue
            anchor = result[shoulder_ids].mean(axis=0)
            result[arm_ids] = anchor + arm_factor * (result[arm_ids] - anchor)
    return result


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# generated by TemplatePattern_final body runtime"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pose_vector(params: dict[str, Any]) -> np.ndarray:
    result = np.zeros(72, dtype=np.float64)
    orient = np.asarray(params.get("global_orient") or [0.0, 0.0, 0.0], dtype=np.float64)
    body_pose = np.asarray(params.get("body_pose") or [], dtype=np.float64)
    result[: min(3, len(orient))] = orient[:3]
    result[3 : 3 + min(69, len(body_pose))] = body_pose[:69]
    return result


@lru_cache(maxsize=4)
def _load_model(path: str) -> dict[str, Any]:
    raw = _load_pickle(Path(path))
    shapedirs = raw["shapedirs"].x if hasattr(raw["shapedirs"], "x") else raw["shapedirs"]
    return {
        "shapedirs": np.asarray(shapedirs, dtype=np.float64),
        "v_template": np.asarray(raw["v_template"], dtype=np.float64),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "posedirs": np.asarray(raw["posedirs"], dtype=np.float64),
        "faces": np.asarray(raw["f"], dtype=np.int64),
        "j_regressor": raw["J_regressor"],
        "parents": smpl_parents(np.asarray(raw["kintree_table"], dtype=np.int64)),
    }


def _load_pickle(path: Path) -> dict[str, Any]:
    class Ch:
        pass

    ch_module = types.ModuleType("chumpy.ch")
    ch_module.Ch = Ch
    chumpy_module = types.ModuleType("chumpy")
    chumpy_module.ch = ch_module
    old = (sys.modules.get("chumpy"), sys.modules.get("chumpy.ch"))
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = ch_module
    try:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")
    finally:
        for name, value in zip(("chumpy", "chumpy.ch"), old):
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def batch_rodrigues(vectors: np.ndarray) -> np.ndarray:
    rotations = []
    for vector in vectors:
        angle = float(np.linalg.norm(vector))
        if angle < 1.0e-12:
            rotations.append(np.eye(3, dtype=np.float64))
            continue
        axis = vector / angle
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        rotations.append(np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * np.matmul(skew, skew))
    return np.stack(rotations)


def smpl_parents(kintree: np.ndarray) -> list[int]:
    mapping = {int(kintree[1, index]): index for index in range(kintree.shape[1])}
    return [-1] + [mapping[int(kintree[0, index])] for index in range(1, kintree.shape[1])]


def global_rigid_transforms(rotations: np.ndarray, joints: np.ndarray, parents: list[int]) -> np.ndarray:
    transforms = [_transform(rotations[0], joints[0])]
    for index in range(1, len(parents)):
        transforms.append(np.matmul(transforms[parents[index]], _transform(rotations[index], joints[index] - joints[parents[index]])))
    return np.stack(transforms)


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def pack(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], 4, 4), dtype=np.float64)
    result[:, :, 3:4] = values
    return result


def _valid_indices(values: list[int], size: int) -> list[int]:
    return sorted({int(value) for value in values if 0 <= int(value) < size})


def _segment_y(vertices: np.ndarray, segmentation: dict[str, list[int]], names: tuple[str, ...], default: float) -> float:
    indices = _valid_indices([index for name in names for index in segmentation.get(name, [])], len(vertices))
    if not indices:
        return default
    y_min = float(vertices[:, 1].min())
    height = max(float(vertices[:, 1].max()) - y_min, 1.0e-8)
    return float((np.median(vertices[indices, 1]) - y_min) / height)
