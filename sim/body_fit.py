from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from .smpl_model import SmplModel


MEASUREMENT_KEYS = ("height", "chest", "waist", "hip", "shoulder_width", "arm_length")
TOLERANCES_CM = {
    "height": 1.5,
    "chest": 2.0,
    "waist": 2.0,
    "hip": 2.0,
    "shoulder_width": 1.5,
    "arm_length": 1.5,
}
SECTION_LEVELS = {"chest": 0.72, "waist": 0.55, "hip": 0.48}


@dataclass(frozen=True)
class BodyFitResult:
    params: dict[str, Any]
    correction: dict[str, Any]
    report: dict[str, Any]


class BodyFitter:
    def __init__(self, model: SmplModel, segmentation: dict[str, list[int]]) -> None:
        self.model = model
        self.segmentation = segmentation
        torso_names = ("hips", "spine", "spine1", "spine2", "neck")
        self.torso_indices = {int(index) for name in torso_names for index in segmentation.get(name, [])}
        # Anatomical section heights are below the arms and above the separated
        # legs. Intersecting the complete watertight mesh avoids open contours
        # caused by segmentation boundaries while retaining a smooth objective.
        self.torso_faces = model.faces

    def fit(self, base_params: dict[str, Any], target: dict[str, float], max_nfev: int = 80) -> BodyFitResult:
        self._validate_target(target)
        initial = np.zeros(10, dtype=np.float64)
        raw_initial = np.asarray(base_params.get("betas") or [], dtype=np.float64)
        initial[: min(10, len(raw_initial))] = raw_initial[:10]

        def evaluate(betas: np.ndarray) -> tuple[float, dict[str, float]]:
            canonical = self.model.canonical_vertices(betas)
            raw_height_cm = float(np.ptp(canonical[:, 1]) * 100.0)
            scale = float(target["height"] / max(raw_height_cm, 1.0e-8))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"target height requires invalid body_scale {scale}")
            return scale, self.measure(canonical * scale * 100.0)

        def residuals(betas: np.ndarray) -> np.ndarray:
            _scale, measured = evaluate(betas)
            values = [(measured[key] - target[key]) / TOLERANCES_CM[key] for key in MEASUREMENT_KEYS[1:]]
            values.extend(0.025 * np.asarray(betas, dtype=np.float64))
            return np.asarray(values, dtype=np.float64)

        result = least_squares(
            residuals,
            initial,
            bounds=(-5.0, 5.0),
            x_scale="jac",
            diff_step=0.03,
            max_nfev=max_nfev,
            ftol=1.0e-7,
            xtol=1.0e-7,
            gtol=1.0e-7,
            verbose=0,
        )
        scale, pure = evaluate(result.x)
        correction, final = self._correct(result.x, scale, target)
        correction["applied"] = any(
            abs(float(value) - 1.0) > 1.0e-8
            for value in (
                *correction["torso_radial"].values(),
                correction["shoulder_width"],
                correction["arm_length"],
            )
        )
        correction["target_measurements_cm"] = dict(target)
        correction["pure_smpl_measurements_cm"] = dict(pure)
        errors = {key: float(final[key] - target[key]) for key in MEASUREMENT_KEYS}
        accepted = all(abs(errors[key]) <= TOLERANCES_CM[key] for key in MEASUREMENT_KEYS)
        params = dict(base_params)
        params.update({"gender": "male", "body_scale": scale, "betas": [float(value) for value in result.x[:10]]})
        report = {
            "method": "smpl_scale_10_betas_plus_bounded_canonical_correction",
            "target_measurements_cm": target,
            "pure_smpl_measurements_cm": pure,
            "final_measurements_cm": final,
            "absolute_errors_cm": {key: abs(value) for key, value in errors.items()},
            "signed_errors_cm": errors,
            "tolerances_cm": dict(TOLERANCES_CM),
            "accepted": accepted,
            "optimizer": {
                "success": bool(result.success),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
            },
        }
        if not accepted:
            failures = ", ".join(
                f"{key}={errors[key]:+.3f}cm (limit {TOLERANCES_CM[key]:.3f}cm)"
                for key in MEASUREMENT_KEYS
                if abs(errors[key]) > TOLERANCES_CM[key]
            )
            raise BodyFitError(f"fitted SMPL body does not meet measurement tolerances: {failures}", params, correction, report)
        return BodyFitResult(params, correction, report)

    def measure(self, vertices_cm: np.ndarray) -> dict[str, float]:
        vertices = np.asarray(vertices_cm, dtype=np.float64)
        height = float(vertices[:, 1].max() - vertices[:, 1].min())
        result = {"height": height}
        y_min = float(vertices[:, 1].min())
        for name, fraction in SECTION_LEVELS.items():
            result[name] = self._section_perimeter(vertices, y_min + height * fraction)
        result["shoulder_width"] = self._shoulder_width(vertices)
        result["arm_length"] = self._arm_length(vertices)
        return result

    def _section_perimeter(self, vertices: np.ndarray, plane_y: float) -> float:
        total = 0.0
        for face in self.torso_faces:
            points = vertices[face]
            intersections: list[np.ndarray] = []
            for left, right in ((0, 1), (1, 2), (2, 0)):
                a, b = points[left], points[right]
                da, db = float(a[1] - plane_y), float(b[1] - plane_y)
                if da == 0.0 and db == 0.0:
                    continue
                if da * db > 0.0 or abs(da - db) < 1.0e-12:
                    continue
                ratio = da / (da - db)
                if -1.0e-9 <= ratio <= 1.0 + 1.0e-9:
                    point = a + ratio * (b - a)
                    if not any(np.linalg.norm(point - existing) < 1.0e-8 for existing in intersections):
                        intersections.append(point)
            if len(intersections) == 2:
                total += float(np.linalg.norm(intersections[0][[0, 2]] - intersections[1][[0, 2]]))
        if total > 1.0:
            return total
        return self._section_fallback(vertices[list(self.torso_indices)], plane_y)

    def _section_fallback(self, vertices: np.ndarray, plane_y: float) -> float:
        points = vertices[np.argsort(np.abs(vertices[:, 1] - plane_y))[:40]][:, [0, 2]]
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        return float(sum(np.linalg.norm(polygon[(index + 1) % len(polygon)] - polygon[index]) for index in range(len(polygon))))

    def _shoulder_width(self, vertices: np.ndarray) -> float:
        left = vertices[_indices(self.segmentation, "leftShoulder", len(vertices))]
        right = vertices[_indices(self.segmentation, "rightShoulder", len(vertices))]
        return float(abs(np.percentile(left[:, 0], 80) - np.percentile(right[:, 0], 20)))

    def _arm_length(self, vertices: np.ndarray) -> float:
        values = []
        for side in ("left", "right"):
            shoulder = _center(vertices, self.segmentation, f"{side}Shoulder")
            arm = _center(vertices, self.segmentation, f"{side}Arm")
            forearm = _center(vertices, self.segmentation, f"{side}ForeArm")
            hand = vertices[_indices(self.segmentation, f"{side}Hand", len(vertices))]
            hand_root = hand[np.argmin(np.linalg.norm(hand - forearm.reshape(1, 3), axis=1))]
            values.append(np.linalg.norm(arm - shoulder) + np.linalg.norm(forearm - arm) + np.linalg.norm(hand_root - forearm))
        return float(np.mean(values))

    def _correct(
        self, betas: np.ndarray, scale: float, target: dict[str, float]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        factors = {"chest": 1.0, "waist": 1.0, "hip": 1.0, "shoulder_width": 1.0, "arm_length": 1.0}
        canonical = self.model.canonical_vertices(betas)
        for _iteration in range(10):
            correction = _correction_payload(factors)
            corrected = self.model.canonical_vertices(betas, correction, self.segmentation) * scale * 100.0
            measured = self.measure(corrected)
            if all(abs(measured[key] - target[key]) <= TOLERANCES_CM[key] for key in MEASUREMENT_KEYS):
                correction["iterations"] = _iteration
                return correction, measured
            for key in factors:
                current = max(float(measured[key]), 1.0e-8)
                updated = factors[key] * float(target[key]) / current
                factors[key] = float(np.clip(updated, 0.85, 1.15))
        correction = _correction_payload(factors)
        correction["iterations"] = 10
        final = self.model.canonical_vertices(betas, correction, self.segmentation) * scale * 100.0
        return correction, self.measure(final)

    def _validate_target(self, target: dict[str, float]) -> None:
        missing = [key for key in MEASUREMENT_KEYS if key not in target]
        if missing:
            raise ValueError(f"body target is missing measurements: {', '.join(missing)}")
        invalid = [key for key in MEASUREMENT_KEYS if not np.isfinite(float(target[key])) or float(target[key]) <= 0.0]
        if invalid:
            raise ValueError(f"body target has invalid measurements: {', '.join(invalid)}")


class BodyFitError(RuntimeError):
    def __init__(self, message: str, params: dict[str, Any], correction: dict[str, Any], report: dict[str, Any]) -> None:
        super().__init__(message)
        self.params = params
        self.correction = correction
        self.report = report


def _correction_payload(factors: dict[str, float]) -> dict[str, Any]:
    return {
        "mode": "bounded_symmetric_canonical",
        "limits": [0.85, 1.15],
        "torso_radial": {key: float(factors[key]) for key in ("chest", "waist", "hip")},
        "shoulder_width": float(factors["shoulder_width"]),
        "arm_length": float(factors["arm_length"]),
    }


def _indices(segmentation: dict[str, list[int]], name: str, size: int) -> list[int]:
    values = sorted({int(index) for index in segmentation.get(name, []) if 0 <= int(index) < size})
    if not values:
        raise ValueError(f"body segmentation has no valid vertices for {name}")
    return values


def _center(vertices: np.ndarray, segmentation: dict[str, list[int]], name: str) -> np.ndarray:
    return vertices[_indices(segmentation, name, len(vertices))].mean(axis=0)
