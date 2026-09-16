from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .body_fit import (
    BodyFitError,
    BodyFitResult,
    BodyFitter,
    MEASUREMENT_KEYS,
    TOLERANCES_CM,
    _correction_payload,
)
from .mhr_model import MhrSmplTopologyModel
from .smpl_model import apply_body_correction


# Three body identity components handle torso girth.  Five explicitly named
# MHR controls handle height distribution, shoulders, arms, hips and legs.
CONTROL_NAMES = (
    "spine_length_flexible",
    "shoulder_width_flexible",
    "arm_length_flexible",
    "hip_width_flexible",
    "leg_length_flexible",
)


class MhrBodyFitter:
    def __init__(self, model: MhrSmplTopologyModel, segmentation: dict[str, list[int]]) -> None:
        self.model = model
        self.segmentation = segmentation
        self.measurements = BodyFitter(model, segmentation)

    def fit(self, target: dict[str, float], max_nfev: int = 80) -> BodyFitResult:
        self.measurements._validate_target(target)

        def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            identity = np.zeros(45, dtype=np.float64)
            identity[:3] = values[:3]
            controls = {name: float(value) for name, value in zip(CONTROL_NAMES, values[3:])}
            return identity, self.model.parameters_with_controls(controls)

        cache: dict[tuple[float, ...], tuple[float, dict[str, float], np.ndarray, np.ndarray]] = {}

        def evaluate(values: np.ndarray) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
            key = tuple(float(value) for value in values)
            if key not in cache:
                identity, parameters = unpack(values)
                canonical = self.model.canonical_vertices(identity, parameters)
                raw_height_cm = float(np.ptp(canonical[:, 1]) * 100.0)
                scale = float(target["height"] / max(raw_height_cm, 1.0e-8))
                measured = self.measurements.measure(canonical * scale * 100.0)
                cache[key] = scale, measured, identity, parameters
            return cache[key]

        def residuals(values: np.ndarray) -> np.ndarray:
            _scale, measured, _identity, _parameters = evaluate(values)
            result = [
                (measured[key] - target[key]) / TOLERANCES_CM[key]
                for key in MEASUREMENT_KEYS[1:]
            ]
            result.extend(0.025 * np.asarray(values, dtype=np.float64))
            return np.asarray(result, dtype=np.float64)

        # A small non-zero start is required because MHR runs in float32 and a
        # relative finite-difference step at exact zero would be rounded away.
        initial = np.full(3 + len(CONTROL_NAMES), 0.01, dtype=np.float64)
        lower = np.asarray([-3.0] * 3 + [-1.5] * len(CONTROL_NAMES))
        upper = np.asarray([3.0] * 3 + [1.5] * len(CONTROL_NAMES))
        result = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            diff_step=0.05,
            max_nfev=max_nfev,
            ftol=1.0e-7,
            xtol=1.0e-7,
            gtol=1.0e-7,
            verbose=0,
        )
        scale, pure, identity, parameters = evaluate(result.x)
        canonical = self.model.canonical_vertices(identity, parameters)
        correction, final = self._correct(canonical, scale, target)
        correction["applied"] = any(
            abs(float(value) - 1.0) > 1.0e-8
            for value in (
                *correction["torso_radial"].values(),
                correction["shoulder_width"],
                correction["arm_length"],
            )
        )
        correction["target_measurements_cm"] = dict(target)
        correction["pure_mhr_measurements_cm"] = dict(pure)

        errors = {key: float(final[key] - target[key]) for key in MEASUREMENT_KEYS}
        accepted = all(abs(errors[key]) <= TOLERANCES_CM[key] for key in MEASUREMENT_KEYS)
        controls = {name: float(value) for name, value in zip(CONTROL_NAMES, result.x[3:])}
        params: dict[str, Any] = {
            "body_model": "mhr",
            "lod": 1,
            "topology": "smpl_6890_barycentric",
            "body_scale": scale,
            "identity_coeffs": [float(value) for value in identity],
            "model_parameters": [float(value) for value in parameters],
            "face_expr_coeffs": [0.0] * 72,
            "optimized_controls": controls,
        }
        report = {
            "method": "mhr_3_identity_5_semantic_controls_to_smpl_topology",
            "target_measurements_cm": target,
            "pure_mhr_measurements_cm": pure,
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
            raise BodyFitError(f"fitted MHR body does not meet measurement tolerances: {failures}", params, correction, report)
        return BodyFitResult(params, correction, report)

    def measure_params(self, params: dict[str, Any], correction: dict[str, Any]) -> dict[str, float]:
        canonical = self.model.canonical_vertices(
            params.get("identity_coeffs", params.get("shape_params", [])) or [],
            params.get("model_parameters", params.get("lbs_model_params", params.get("mhr_model_params", []))) or [],
            correction,
            self.segmentation,
        )
        return self.measurements.measure(canonical * float(params.get("body_scale") or 1.0) * 100.0)

    def _correct(
        self, canonical: np.ndarray, scale: float, target: dict[str, float]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        factors = {"chest": 1.0, "waist": 1.0, "hip": 1.0, "shoulder_width": 1.0, "arm_length": 1.0}
        for iteration in range(10):
            correction = _correction_payload(factors)
            corrected = apply_body_correction(canonical, correction, self.segmentation) * scale * 100.0
            measured = self.measurements.measure(corrected)
            if all(abs(measured[key] - target[key]) <= TOLERANCES_CM[key] for key in MEASUREMENT_KEYS):
                correction["iterations"] = iteration
                return correction, measured
            for key in factors:
                current = max(float(measured[key]), 1.0e-8)
                factors[key] = float(np.clip(factors[key] * float(target[key]) / current, 0.85, 1.15))
        correction = _correction_payload(factors)
        correction["iterations"] = 10
        final = apply_body_correction(canonical, correction, self.segmentation) * scale * 100.0
        return correction, self.measurements.measure(final)
