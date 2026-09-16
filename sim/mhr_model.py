from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from TemplatePattern_final.shared.io import ROOT

from .smpl_model import apply_body_correction


MHR_MODEL_FILENAME = "mhr_model.pt"
MHR_TO_SMPL_MAPPING = ROOT / "assets" / "mhr" / "mhr2smpl_mapping.npz"

# MHR's neutral rig has the arms down and forward, with an approximately
# 145-degree natural elbow bend.  Garment initialization needs a straight arm
# axis so sleeves and cuffs are not placed across the inside of that bend.
# These values straighten both elbows and align shoulder-to-wrist with the
# requested garment poses while keeping both sides exactly symmetric.  They
# were solved against the MHR v1.0.1 skeleton.
MHR_STRAIGHT_ELBOW_BEND = -0.6024013
ARM_POSE_PARAMETERS = {
    "tpose": (0.7533271, 0.4216992),
    "a30": (0.3083025, 0.0966706),
    "a45": (0.0752869, -0.0251317),
    "a60": (-0.1587069, -0.1429349),
}


class MhrSmplTopologyModel:
    """MHR LOD1 geometry exposed on the standard 6890-vertex SMPL topology.

    MHR remains the geometry and pose generator.  The final surface is sampled
    with Meta's precomputed MHR-to-SMPL barycentric map so the existing body
    segmentation, placement code and GarmentCode collision filters remain
    valid.  This is intentionally not a lossy refit to ten SMPL betas.
    """

    def __init__(self, model_dir: str | Path, topology_obj: str | Path) -> None:
        self.model_path = Path(model_dir) / MHR_MODEL_FILENAME
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MHR TorchScript model is missing: {self.model_path}. "
                "Download the official MHR assets and place assets/mhr_model.pt "
                "in this directory."
            )
        if not MHR_TO_SMPL_MAPPING.is_file():
            raise FileNotFoundError(f"MHR-to-SMPL topology map is missing: {MHR_TO_SMPL_MAPPING}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("MHR body generation requires PyTorch and Python >= 3.11") from exc

        self.torch = torch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.load` is deprecated")
            self.runtime = torch.jit.load(str(self.model_path), map_location="cpu").eval()

        mhr_faces = self.runtime.character_torch.mesh.faces.detach().cpu().numpy().astype(np.int64)
        mapping = np.load(MHR_TO_SMPL_MAPPING)
        triangle_ids = np.asarray(mapping["triangle_ids"], dtype=np.int64)
        if triangle_ids.shape != (6890,) or int(triangle_ids.max()) >= len(mhr_faces):
            raise ValueError("MHR-to-SMPL mapping is incompatible with the loaded MHR model")
        self._source_triangles = mhr_faces[triangle_ids]
        self._barycentric = np.asarray(mapping["baryc_coords"], dtype=np.float64)
        self._faces = _read_obj_faces(Path(topology_obj))
        if int(self._faces.max()) >= len(triangle_ids):
            raise ValueError(f"SMPL topology OBJ is incompatible with the MHR mapping: {topology_obj}")

        names = list(self.runtime.character_torch.parameter_transform.parameter_names)
        if len(names) < 204:
            raise ValueError("MHR model exposes fewer than the expected 204 model parameters")
        self.parameter_names = names[:204]
        self.parameter_index = {name: index for index, name in enumerate(self.parameter_names)}

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    def canonical_vertices(
        self,
        identity_coeffs: list[float] | np.ndarray,
        model_parameters: list[float] | np.ndarray | None = None,
        correction: dict[str, Any] | None = None,
        segmentation: dict[str, list[int]] | None = None,
    ) -> np.ndarray:
        parameters = self._body_shape_parameters(model_parameters)
        self._set_arm_pose(parameters, "tpose")
        vertices = self._forward(identity_coeffs, parameters)
        if correction:
            if not segmentation:
                raise ValueError("segmentation is required when applying body correction")
            vertices = apply_body_correction(vertices, correction, segmentation)
        return vertices

    def posed_vertices(
        self,
        params: dict[str, Any],
        correction: dict[str, Any] | None,
        segmentation: dict[str, list[int]],
        pose: str,
        arm_angle_deg: float | None = None,
    ) -> np.ndarray:
        parameters = self._body_shape_parameters(
            params.get("model_parameters", params.get("lbs_model_params", params.get("mhr_model_params")))
        )
        if arm_angle_deg is None:
            self._set_arm_pose(parameters, pose)
        else:
            self._set_arm_angle(parameters, float(arm_angle_deg))
        vertices = self._forward(
            params.get("identity_coeffs", params.get("shape_params", [])) or [],
            parameters,
            params.get("face_expr_coeffs", params.get("expr_params", [])) or [],
        )
        if correction:
            vertices = apply_body_correction(vertices, correction, segmentation)
        vertices *= float(params.get("body_scale") or 1.0)
        vertices[:, 1] -= float(vertices[:, 1].min())
        translation = np.asarray(params.get("transl") or [0.0, 0.0, 0.0], dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError("MHR params transl must contain exactly 3 values")
        vertices += translation.reshape(1, 3)
        if not np.isfinite(vertices).all():
            raise ValueError("MHR generation produced non-finite vertices")
        return vertices

    def parameters_with_controls(self, controls: dict[str, float] | None = None) -> np.ndarray:
        parameters = np.zeros(204, dtype=np.float64)
        for name, value in (controls or {}).items():
            if name not in self.parameter_index:
                raise ValueError(f"unknown MHR model parameter: {name}")
            parameters[self.parameter_index[name]] = float(value)
        return parameters

    def _forward(
        self,
        identity_coeffs: list[float] | np.ndarray,
        model_parameters: np.ndarray,
        face_expr_coeffs: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        identity = np.zeros(45, dtype=np.float32)
        source = np.asarray(identity_coeffs, dtype=np.float32).reshape(-1)
        identity[: min(45, len(source))] = source[:45]
        expression = np.zeros(72, dtype=np.float32)
        expression_source = np.asarray(face_expr_coeffs if face_expr_coeffs is not None else [], dtype=np.float32).reshape(-1)
        expression[: min(72, len(expression_source))] = expression_source[:72]
        with self.torch.no_grad():
            vertices, _skeleton = self.runtime(
                self.torch.from_numpy(identity[None, :]),
                self.torch.from_numpy(np.asarray(model_parameters, dtype=np.float32)[None, :]),
                self.torch.from_numpy(expression[None, :]),
            )
        source_vertices = vertices[0].detach().cpu().numpy().astype(np.float64)
        triangles = source_vertices[self._source_triangles]
        # MHR uses centimetres; all body model APIs in this project use metres.
        return (triangles * self._barycentric[:, :, None]).sum(axis=1) / 100.0

    def _parameter_vector(self, values: list[float] | np.ndarray | None) -> np.ndarray:
        result = np.zeros(204, dtype=np.float64)
        source = np.asarray(values if values is not None else [], dtype=np.float64).reshape(-1)
        result[: min(204, len(source))] = source[:204]
        return result

    def _body_shape_parameters(self, values: list[float] | np.ndarray | None) -> np.ndarray:
        parameters = self._parameter_vector(values)
        # Stage 2 owns the pose.  Keep MHR's six flexible proportions and all
        # scale parameters, but discard root/body/hand pose from imported MHR
        # or SAM3D files before applying a30/a45/a60/tpose.
        parameters[:130] = 0.0
        return parameters

    def _set_arm_pose(self, parameters: np.ndarray, pose: str) -> None:
        if pose not in ARM_POSE_PARAMETERS:
            raise ValueError(f"unsupported MHR pose: {pose}")
        upper_y, upper_z = ARM_POSE_PARAMETERS[pose]
        self._write_arm_parameters(parameters, upper_y, upper_z)

    def _set_arm_angle(self, parameters: np.ndarray, angle_deg: float) -> None:
        if not 0.0 <= angle_deg <= 60.0:
            raise ValueError("MHR arm angle must be between 0 and 60 degrees")
        degrees = np.asarray([0.0, 30.0, 45.0, 60.0])
        values = np.asarray([
            ARM_POSE_PARAMETERS["tpose"],
            ARM_POSE_PARAMETERS["a30"],
            ARM_POSE_PARAMETERS["a45"],
            ARM_POSE_PARAMETERS["a60"],
        ])
        upper_y = float(np.interp(angle_deg, degrees, values[:, 0]))
        upper_z = float(np.interp(angle_deg, degrees, values[:, 1]))
        self._write_arm_parameters(parameters, upper_y, upper_z)

    def _write_arm_parameters(self, parameters: np.ndarray, upper_y: float, upper_z: float) -> None:
        for side in ("r", "l"):
            parameters[self.parameter_index[f"{side}_uparm_ry"]] = upper_y
            parameters[self.parameter_index[f"{side}_uparm_rz"]] = upper_z
            parameters[self.parameter_index[f"{side}_elbow_bend"]] = MHR_STRAIGHT_ELBOW_BEND


def default_mhr_model_dir(configured: str | Path | None = None) -> Path:
    return Path(configured or os.environ.get("TEMPLATEPATTERN_MHR_MODEL_DIR") or ROOT / "models" / "mhr")


def _read_obj_faces(path: Path) -> np.ndarray:
    faces: list[list[int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("f "):
            continue
        values = [int(token.split("/", 1)[0]) - 1 for token in raw.split()[1:]]
        if len(values) != 3:
            raise ValueError(f"body topology must be triangulated: {path}")
        faces.append(values)
    if not faces:
        raise ValueError(f"body topology OBJ has no faces: {path}")
    return np.asarray(faces, dtype=np.int64)
