from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
from typing import Any

from TemplatePattern_final.shared.body_config import BodyConfig
from TemplatePattern_final.shared.io import ROOT, read_json, write_json

from .body_fit import BodyFitError, BodyFitter, MEASUREMENT_KEYS, TOLERANCES_CM
from .mhr_fit import MhrBodyFitter
from .mhr_model import MhrSmplTopologyModel, default_mhr_model_dir
from .smpl_model import SmplModel, write_obj


BASE_PARAMS = {
    "long_sleeve": "torso_v6_male_asia_M_a45_smpl_params.json",
    "short_sleeve": "torso_short_v6_male_asia_M_a45_smpl_params.json",
}
BASE_YAML = {
    "long_sleeve": "torso_v6_male_asia_M_a45_smpl.yaml",
    "short_sleeve": "torso_short_v6_male_asia_M_a45_smpl.yaml",
}


class BodyAssetManager:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        body_model: str = "mhr",
        mhr_model_dir: str | Path | None = None,
    ) -> None:
        if body_model not in ("mhr", "smpl"):
            raise ValueError(f"unsupported body model: {body_model}")
        self.body_model = body_model
        configured = model_dir or os.environ.get("TEMPLATEPATTERN_SMPL_MODEL_DIR") or ROOT / "models" / "smpl"
        self.model_dir = Path(configured).resolve()
        self.mhr_model_dir = default_mhr_model_dir(mhr_model_dir).resolve()

    def prepare_assets(
        self,
        style: str,
        out_dir: str | Path,
        *,
        pose: str = "a30",
        arm_angle_deg: float = 30.0,
        body_config: str | Path | None = None,
        body_params: str | Path | None = None,
        body_target: str | Path | None = None,
        pattern_json: str | Path | None = None,
    ) -> dict[str, Any]:
        source = self._source_dir(style)
        target_dir = Path(out_dir) / "body"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("smpl_vert_segmentation.json", "ggg_body_segmentation.json"):
            shutil.copy2(source / name, target_dir / name)
        segmentation = read_json(source / "smpl_vert_segmentation.json")
        if self.body_model == "mhr":
            topology_obj = source / (
                "torso_short_v6_male_asia_M_tpose_smpl.obj"
                if style == "short_sleeve"
                else "torso_v6_tpose_male_asia_M_smpl.obj"
            )
            model: Any = MhrSmplTopologyModel(self.mhr_model_dir, topology_obj)
        else:
            model = SmplModel(self.model_dir)
        try:
            params, correction, report, source_info = self._fit_or_load(
                model, segmentation, source, style, body_config, body_params, body_target, pattern_json
            )
        except BodyFitError as exc:
            write_json(target_dir / ("mhr_params.json" if self.body_model == "mhr" else "smpl_params.json"), exc.params)
            write_json(target_dir / "body_correction.json", exc.correction)
            write_json(target_dir / "body_measurement_report.json", exc.report)
            raise RuntimeError(f"dynamic {self.body_model.upper()} fitting failed: {exc}") from exc
        params["stage2_pose"] = pose
        params["arm_angle_deg"] = float(arm_angle_deg)
        if self.body_model == "mhr":
            vertices = model.posed_vertices(params, correction, segmentation, pose, arm_angle_deg)
        else:
            params["body_pose"] = [0.0] * 69 if pose == "tpose" else self._a_pose_body_pose(params.get("body_pose"), arm_angle_deg)
            vertices = model.posed_vertices(params, correction, segmentation)
        write_obj(target_dir / "body.obj", vertices, model.faces)
        self._write_body_yaml(target_dir / "body.yaml", source / BASE_YAML[style], params, report)
        if self.body_model == "mhr":
            write_json(target_dir / "mhr_params.json", params)
            write_json(target_dir / "smpl_params.json", self._smpl_compatibility_params(params))
        else:
            write_json(target_dir / "smpl_params.json", params)
        write_json(target_dir / "body_correction.json", correction)
        write_json(target_dir / "body_measurement_report.json", report)
        write_json(target_dir / "body_source.json", source_info)
        return {
            "body_assets_dir": str(target_dir),
            "body_name": "body",
            "body_model": self.body_model,
            "body_params": str(target_dir / ("mhr_params.json" if self.body_model == "mhr" else "smpl_params.json")),
            "mhr_params": str(target_dir / "mhr_params.json") if self.body_model == "mhr" else None,
            "body_correction": str(target_dir / "body_correction.json"),
            "body_measurement_report": str(target_dir / "body_measurement_report.json"),
            "body_source": source_info["source"],
            "body_input_mm": source_info.get("body_input_mm"),
            "effective_body_input_mm": source_info.get("body_input_mm"),
            "requested_body_input_mm": source_info.get("requested_body_input_mm", source_info.get("body_input_mm")),
            "input_resolution": source_info.get("input_resolution"),
            "target_measurements_cm": report.get("target_measurements_cm"),
            "fit_report": report,
            "body_source_debug": str(target_dir / "body_source.json"),
        }

    def _fit_or_load(
        self,
        model: Any,
        segmentation: dict[str, list[int]],
        source: Path,
        style: str,
        body_config: str | Path | None,
        body_params: str | Path | None,
        body_target: str | Path | None,
        pattern_json: str | Path | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if body_params:
            params = read_json(body_params)
            if self.body_model == "mhr":
                params = self._normalize_mhr_params(params)
            target_info = self._body_target(body_target, body_config, pattern_json)
            target_cm = {key: float(target_info["body_input_mm"][key]) / 10.0 for key in MEASUREMENT_KEYS}
            correction_path = Path(body_params).with_name("body_correction.json")
            correction = read_json(correction_path) if correction_path.exists() else self._identity_correction()
            if self.body_model == "mhr":
                if "body_scale" not in params:
                    canonical = model.canonical_vertices(
                        params.get("identity_coeffs") or [], params.get("model_parameters") or []
                    )
                    raw_height_cm = float((canonical[:, 1].max() - canonical[:, 1].min()) * 100.0)
                    params["body_scale"] = target_cm["height"] / max(raw_height_cm, 1.0e-8)
                measured = MhrBodyFitter(model, segmentation).measure_params(params, correction)
                method = "provided_mhr_params_to_smpl_topology"
            else:
                canonical = model.canonical_vertices(params.get("betas") or [], correction, segmentation)
                measured = BodyFitter(model, segmentation).measure(canonical * float(params.get("body_scale") or 1.0) * 100.0)
                method = "provided_smpl_params"
            errors = {key: float(measured[key] - target_cm[key]) for key in MEASUREMENT_KEYS}
            accepted = all(abs(errors[key]) <= TOLERANCES_CM[key] for key in MEASUREMENT_KEYS)
            report = {
                "method": method,
                "target_measurements_cm": target_cm,
                "final_measurements_cm": measured,
                "absolute_errors_cm": {key: abs(value) for key, value in errors.items()},
                "signed_errors_cm": errors,
                "tolerances_cm": dict(TOLERANCES_CM),
                "accepted": accepted,
            }
            source_info = {
                "source": "body_params",
                "sample_name": target_info["sample_name"],
                "body_input_mm": target_info["body_input_mm"],
                "requested_body_input_mm": target_info.get("requested_body_input_mm", target_info["body_input_mm"]),
                "input_resolution": target_info.get("input_resolution"),
            }
            if not accepted:
                failures = ", ".join(
                    f"{key}={errors[key]:+.3f}cm (limit {TOLERANCES_CM[key]:.3f}cm)"
                    for key in MEASUREMENT_KEYS
                    if abs(errors[key]) > TOLERANCES_CM[key]
                )
                raise BodyFitError(f"provided {self.body_model.upper()} parameters do not match Stage1 target: {failures}", params, correction, report)
            return params, correction, report, source_info

        target_info = self._body_target(body_target, body_config, pattern_json)
        target_cm = {key: float(target_info["body_input_mm"][key]) / 10.0 for key in MEASUREMENT_KEYS}
        if self.body_model == "mhr":
            fitted = MhrBodyFitter(model, segmentation).fit(target_cm)
        else:
            base = deepcopy(read_json(source / BASE_PARAMS[style]))
            base["gender"] = "male"
            fitted = BodyFitter(model, segmentation).fit(base, target_cm)
        params = fitted.params
        params.update(
            {
                "sample_name": target_info["sample_name"],
                "body_input_mm": target_info["body_input_mm"],
                "target_measurements_cm": target_cm,
            }
        )
        source_info = {
            "source": target_info["source"],
            "sample_name": target_info["sample_name"],
            "body_input_mm": target_info["body_input_mm"],
            "requested_body_input_mm": target_info.get("requested_body_input_mm", target_info["body_input_mm"]),
            "input_resolution": target_info.get("input_resolution"),
        }
        return params, fitted.correction, fitted.report, source_info

    def _body_target(
        self,
        body_target: str | Path | None,
        body_config: str | Path | None,
        pattern_json: str | Path | None,
    ) -> dict[str, Any]:
        if body_target:
            path = Path(body_target)
            data = read_json(path)
            body = BodyConfig.load(path)
            return {
                "source": "stage1_body_target",
                "sample_name": body.sample_name,
                "body_input_mm": body.body_input,
                "requested_body_input_mm": {
                    key: float((data.get("requested_body_input") or body.body_input)[key])
                    for key in MEASUREMENT_KEYS
                },
                "input_resolution": data.get("input_resolution"),
                "schema_version": data.get("schema_version"),
            }
        if body_config:
            body = BodyConfig.load(body_config)
            return {
                "source": "body_config",
                "sample_name": body.sample_name,
                "body_input_mm": body.body_input,
                "requested_body_input_mm": body.body_input,
                "input_resolution": {"mode": "requested_input", "adjusted": False},
            }
        pattern = Path(pattern_json) if pattern_json else None
        if not pattern:
            raise FileNotFoundError("Stage2 needs --body-target, --body-config, or a pattern inside a Stage1 task directory")
        body_target_path = pattern.parent / "body_target.json"
        if not body_target_path.exists():
            raise FileNotFoundError(f"Stage2 body target is missing: {body_target_path}")
        return self._body_target(body_target_path, None, None)

    def _source_dir(self, style: str) -> Path:
        return ROOT / "assets" / ("body_short" if style == "short_sleeve" else "body_long")

    def _a_pose_body_pose(self, original: Any, arm_angle_deg: float) -> list[float]:
        values = [0.0] * 69
        if isinstance(original, list):
            values[: min(69, len(original))] = [float(value) for value in original[:69]]
        angle = math.pi * float(arm_angle_deg) / 180.0
        values[(16 - 1) * 3 + 2] = -angle
        values[(17 - 1) * 3 + 2] = angle
        return values

    def _identity_correction(self) -> dict[str, Any]:
        return {
            "mode": "identity",
            "limits": [0.85, 1.15],
            "torso_radial": {"chest": 1.0, "waist": 1.0, "hip": 1.0},
            "shoulder_width": 1.0,
            "arm_length": 1.0,
        }

    def _write_body_yaml(self, path: Path, source_yaml: Path, params: dict[str, Any], report: dict[str, Any]) -> None:
        measurements = self._read_body_measurements(source_yaml)
        target = report.get("target_measurements_cm") or report.get("final_measurements_cm") or {}
        mapping = {"height": "height", "chest": "bust", "waist": "waist", "hip": "hips", "shoulder_width": "shoulder_w", "arm_length": "arm_length"}
        for source_key, output_key in mapping.items():
            if source_key in target:
                measurements[output_key] = float(target[source_key])
        lines = [
            "body:",
            "  body_sample: body",
            f"  generated_from: {'mhr_lod1' if self.body_model == 'mhr' else 'basic_male_smpl'}",
            f"  generation_mode: {'mhr_to_smpl_topology_with_bounded_correction' if self.body_model == 'mhr' else 'smpl_lbs_with_bounded_correction'}",
            f"  body_scale: {float(params.get('body_scale') or 1.0):.8g}",
        ]
        lines.extend(f"  {key}: {float(value):.8g}" for key, value in sorted(measurements.items()))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_body_measurements(self, path: Path) -> dict[str, float]:
        result: dict[str, float] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0]
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            try:
                result[key] = float(value)
            except ValueError:
                pass
        return result

    def _smpl_compatibility_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Keep the historical filename without claiming a ten-beta SMPL fit."""
        return {
            "body_model": "mhr",
            "conversion_mode": "mhr_surface_to_smpl_6890_barycentric",
            "topology": "smpl_6890",
            "mhr_params": "mhr_params.json",
            "body_scale": float(params.get("body_scale") or 1.0),
            "stage2_pose": params.get("stage2_pose"),
            "arm_angle_deg": params.get("arm_angle_deg"),
            "betas": None,
        }

    def _normalize_mhr_params(self, source: dict[str, Any]) -> dict[str, Any]:
        """Accept native MHR and common SAM3D naming, then store one schema."""
        params = dict(source)
        identity = params.get("identity_coeffs", params.get("shape_params", []))
        model = params.get(
            "model_parameters",
            params.get("lbs_model_params", params.get("mhr_model_params", [])),
        )
        expression = params.get("face_expr_coeffs", params.get("expr_params", [0.0] * 72))

        def flat(values: Any) -> list[float]:
            array = values
            if isinstance(array, list) and len(array) == 1 and isinstance(array[0], list):
                array = array[0]
            return [float(value) for value in array]

        identity_values = flat(identity)
        model_values = flat(model)
        expression_values = flat(expression)
        if len(identity_values) != 45:
            raise ValueError(f"MHR identity/shape parameters must contain 45 values, got {len(identity_values)}")
        if len(model_values) != 204:
            raise ValueError(f"MHR model parameters must contain 204 values, got {len(model_values)}")
        if len(expression_values) != 72:
            raise ValueError(f"MHR face expression parameters must contain 72 values, got {len(expression_values)}")
        params.update({
            "body_model": "mhr",
            "lod": 1,
            "topology": "smpl_6890_barycentric",
            "identity_coeffs": identity_values,
            "model_parameters": model_values,
            "face_expr_coeffs": expression_values,
        })
        return params
