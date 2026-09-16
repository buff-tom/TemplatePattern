from __future__ import annotations

from pathlib import Path
import zipfile
from typing import Any

from TemplatePattern_final.shared.io import ROOT, read_json, write_json
from TemplatePattern_final.shared.task_bundle import Stage1Bundle

from .body_assets import BodyAssetManager
from .garmentcode_runner import GarmentCodeRunner
from .spec_preview import write_spec_preview
from .spec_converter import GarmentSpecConverter
from .tpose_placement import BodyPosePlacement


RUN_NAMES = {(style, pose): f"{style}_{pose}" for style in ("long_sleeve", "short_sleeve") for pose in ("a30", "a45", "a60", "tpose")}
DEFAULT_ARM_ANGLES = {"a30": 30.0, "a60": 60.0, "a45": 45.0, "tpose": 0.0}


class SimulationStage2Pipeline:
    def __init__(
        self,
        style: str | None = None,
        pattern_json: str | Path | None = None,
        output_dir: str | Path | None = None,
        garmentcode_root: str | Path | None = None,
        warp_root: str | Path | None = None,
        sim_config: str | Path | None = None,
        *,
        stage1_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        mhr_model_dir: str | Path | None = None,
        body_model: str = "mhr",
        arm_angle_deg: float | None = None,
        body_config: str | Path | None = None,
        body_params: str | Path | None = None,
        body_target: str | Path | None = None,
        pose: str = "a30",
        preserve_old_output: bool = False,
    ) -> None:
        del preserve_old_output
        self.bundle = Stage1Bundle.load(stage1_dir) if stage1_dir else None
        self.style = self.bundle.style if self.bundle else str(style or "")
        if self.style not in ("long_sleeve", "short_sleeve"):
            raise ValueError(f"invalid Stage2 style: {self.style!r}")
        self.pose = pose
        self.arm_angle_deg = float(DEFAULT_ARM_ANGLES[pose] if arm_angle_deg is None else arm_angle_deg)
        self.body_config = Path(body_config) if body_config else None
        self.body_params = Path(body_params).resolve() if body_params else None
        self.body_target = self.bundle.body_target if self.bundle else Path(body_target) if body_target else None
        self.name = RUN_NAMES[(self.style, pose)]
        self.pattern_json = self.bundle.pattern if self.bundle else Path(pattern_json) if pattern_json else self._default_pattern_json()
        self.run_dir = (Path(output_dir) if output_dir else ROOT / "outputs" / self.style / "stage2").resolve()
        self.body_manager = BodyAssetManager(model_dir, body_model=body_model, mhr_model_dir=mhr_model_dir)
        self.runner = GarmentCodeRunner(garmentcode_root, warp_root, sim_config)

    def run(self, *, convert_only: bool = False, boxmesh_only: bool = False, max_sim_steps: int | None = None, archive: bool = False, draco: bool | None = None) -> dict[str, Any]:
        draco = (not convert_only and not boxmesh_only) if draco is None else draco
        try:
            if draco:
                if convert_only or boxmesh_only:
                    raise ValueError('--draco requires a complete simulation')
                import importlib.util
                if importlib.util.find_spec('DracoPy') is None:
                    raise RuntimeError('Draco export requires DracoPy==1.7.0; install it before simulation')
            if not convert_only and not boxmesh_only:
                from .render_support import preflight
                preflight()
            result = self._run(convert_only=convert_only, boxmesh_only=boxmesh_only, max_sim_steps=max_sim_steps, archive=False)
            if draco:
                if convert_only or boxmesh_only:
                    raise ValueError('--draco requires a complete simulation')
                from .draco_export import export_glb
                result['draco'] = export_glb(self.run_dir / 'simulation/sim.obj', self.run_dir / 'simulation/scene_draco.glb', self.run_dir / 'body/body.obj')
                manifest = read_json(self.run_dir / 'stage2_manifest.json')
                manifest['outputs']['draco_glb'] = 'simulation/scene_draco.glb'
                write_json(self.run_dir / 'stage2_manifest.json', manifest)
            if archive:
                result['archive'] = str(self.create_archive())
            return result
        except BaseException as exc:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            from .garmentcode_runner import SimulationQualityError
            export_error = None
            if draco and isinstance(exc, SimulationQualityError):
                try:
                    from .draco_export import export_glb
                    export_glb(self.run_dir/'simulation/sim.obj', self.run_dir/'simulation/scene_draco.glb', self.run_dir/'body/body.obj')
                except Exception as export_exc:
                    export_error = str(export_exc)
            manifest = {
                'schema_version': 1, 'stage': 'stage2', 'status': 'failed',
                'style': self.style, 'pose': self.pose,
                'body_model': self.body_manager.body_model,
                **self._body_target_manifest_fields(),
                'error': f'{type(exc).__name__}: {exc}',
                'outputs': {str(p.relative_to(self.run_dir)): str(p.relative_to(self.run_dir))
                            for p in self.run_dir.rglob('*') if p.is_file() and p.name not in ('stage2_manifest.json', 'stage2_results.zip')},
            }
            if export_error:
                manifest['draco_export_error'] = export_error
            if draco and isinstance(exc, SimulationQualityError) and not export_error:
                manifest['outputs']['draco_glb'] = 'simulation/scene_draco.glb'
                manifest['draco_quality'] = 'simulation_failed_quality_checks'
            write_json(self.run_dir / 'stage2_manifest.json', manifest)
            if archive:
                self.create_archive()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(f"{exc}; failure manifest: {self.run_dir / 'stage2_manifest.json'}") from exc

    def _run(self, *, convert_only=False, boxmesh_only=False, max_sim_steps=None, archive=False):
        input_dir = self.run_dir / "input"
        simulation_dir = self.run_dir / "simulation"
        input_dir.mkdir(parents=True, exist_ok=True)
        simulation_dir.mkdir(parents=True, exist_ok=True)
        spec_path = input_dir / "garment_specification.json"
        GarmentSpecConverter().convert_file(self.pattern_json, spec_path, self.name)
        assets = self.body_manager.prepare_assets(
            self.style,
            self.run_dir,
            pose=self.pose,
            arm_angle_deg=self.arm_angle_deg,
            body_config=self.body_config,
            body_params=self.body_params,
            body_target=self.body_target,
            pattern_json=self.pattern_json,
        )
        import yaml
        sim_settings = yaml.safe_load(self.runner.sim_config.read_text())['sim']['config']
        wrist_clearance = (float(sim_settings['options'].get('body_collision_thickness', 0.25))
                           + float(sim_settings['material'].get('fabric_thickness', 0.1)) + 0.1)
        placement = BodyPosePlacement().apply_file(
            spec_path,
            self.style,
            body_assets_dir=assets["body_assets_dir"],
            body_name="body",
            pose=self.pose,
            arm_angle_deg=self.arm_angle_deg,
            wrist_clearance_cm=wrist_clearance,
        )
        write_spec_preview(spec_path, simulation_dir / "pattern.svg")
        engine = self.runner.run(
            "garment",
            self.run_dir,
            "body",
            assets["body_assets_dir"],
            convert_only=convert_only,
            boxmesh_only=boxmesh_only,
            max_sim_steps=max_sim_steps,
        )
        stop_after = "convert" if convert_only else "boxmesh" if boxmesh_only else "simulation"
        manifest = self._manifest(assets, placement, engine, stop_after)
        manifest_path = self.run_dir / "stage2_manifest.json"
        write_json(manifest_path, manifest)
        archive_path = self.create_archive() if archive else None
        outputs = manifest["outputs"]
        return {
            "status": "completed",
            "stage2_dir": str(self.run_dir.resolve()),
            "manifest": str(manifest_path.resolve()),
            "archive": str(archive_path) if archive_path else None,
            "boxmesh": self._absolute(outputs.get("boxmesh")),
            "simulation": self._absolute(outputs.get("simulation")),
            "render_front": self._absolute(outputs.get("render_front")),
            "render_back": self._absolute(outputs.get("render_back")),
        }

    def create_archive(self) -> Path:
        """Package the complete task with relative names; exclude the archive itself."""
        destination = self.run_dir / "stage2_results.zip"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(self.run_dir.rglob("*")):
                if source.is_file() and source != destination:
                    archive.write(source, source.relative_to(self.run_dir))
        return destination

    def _manifest(
        self,
        assets: dict[str, Any],
        placement: dict[str, str],
        engine: dict[str, Any],
        stop_after: str,
    ) -> dict[str, Any]:
        outputs = {
            "garment_specification": "input/garment_specification.json",
            "conversion_debug": "input/conversion_debug.json",
            "body_pose_relation": "input/body_pose_relation.json",
            "body_obj": "body/body.obj",
            "body_yaml": "body/body.yaml",
            "smpl_params": "body/smpl_params.json",
            "mhr_params": self._existing_relative("body/mhr_params.json"),
            "body_correction": "body/body_correction.json",
            "body_measurement_report": "body/body_measurement_report.json",
            "placement_landmarks": "body/placement_landmarks.json",
            "boxmesh": self._existing_relative("simulation/boxmesh.obj"),
            "simulation": self._existing_relative("simulation/sim.obj"),
            "render_front": self._existing_relative("simulation/render_front.png"),
            "render_back": self._existing_relative("simulation/render_back.png"),
        }
        return {
            "schema_version": 1,
            "stage": "stage2",
            "status": "completed",
            "stop_after": stop_after,
            "style": self.style,
            "pose": self.pose,
            "arm_angle_deg": self.arm_angle_deg,
            "body_model": assets.get("body_model"),
            "stage1_digest": self.bundle.digest() if self.bundle else None,
            "body_source": assets.get("body_source"),
            "body_input_mm": assets.get("body_input_mm"),
            "effective_body_input_mm": assets.get("effective_body_input_mm", assets.get("body_input_mm")),
            "requested_body_input_mm": assets.get("requested_body_input_mm", assets.get("body_input_mm")),
            "input_resolution": assets.get("input_resolution"),
            "measurement_accepted": bool((assets.get("fit_report") or {}).get("accepted", True)),
            "outputs": outputs,
            "engine": {"pipeline": engine.get("pipeline"), "sim_config": Path(engine["sim_config"]).name},
        }

    def _default_pattern_json(self) -> Path:
        return ROOT / "outputs" / self.style / "stage1" / "pattern.json"

    def _existing_relative(self, value: str) -> str | None:
        return value if (self.run_dir / value).exists() else None

    def _absolute(self, value: str | None) -> str | None:
        return str((self.run_dir / value).resolve()) if value else None

    def _body_target_manifest_fields(self) -> dict[str, Any]:
        if not self.body_target or not self.body_target.is_file():
            return {}
        try:
            target = read_json(self.body_target)
            if not isinstance(target, dict):
                return {}
            effective = target.get("effective_body_input", target.get("body_input"))
            requested = target.get("requested_body_input", effective)
            return {
                "body_input_mm": effective,
                "effective_body_input_mm": effective,
                "requested_body_input_mm": requested,
                "input_resolution": target.get("input_resolution"),
            }
        except (OSError, ValueError, TypeError):
            return {}
