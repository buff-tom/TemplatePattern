from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from TemplatePattern_final.shared.io import ROOT, write_json


class GarmentCodeRunner:
    def __init__(
        self,
        garmentcode_root: str | Path | None = None,
        warp_root: str | Path | None = None,
        sim_config: str | Path | None = None,
    ) -> None:
        garmentcode_default = os.environ.get("TEMPLATEPATTERN_GARMENTCODE_ROOT") or ROOT.parent / "GarmentCode"
        warp_default = os.environ.get("TEMPLATEPATTERN_WARP_ROOT") or ROOT.parent / "NvidiaWarp-GarmentCode"
        self.garmentcode_root = Path(garmentcode_root or garmentcode_default).resolve()
        self.warp_root = Path(warp_root or warp_default).resolve()
        self.sim_config = Path(sim_config or ROOT / "assets" / "garmentcode_sim_props" / "default_sim_props.yaml").resolve()

    def run(
        self,
        name: str,
        run_dir: str | Path,
        body_name: str,
        body_assets_dir: str | Path,
        *,
        boxmesh_only: bool = False,
        convert_only: bool = False,
        max_sim_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(run_dir).resolve()
        spec_path = run_dir / "input" / f"{name}_specification.json"
        if convert_only:
            return self._manifest(name, run_dir, spec_path, body_name, body_assets_dir, None, metadata)
        paths = self._simulate(name, run_dir, spec_path, body_name, body_assets_dir, boxmesh_only, max_sim_steps)
        return self._manifest(name, run_dir, spec_path, body_name, body_assets_dir, paths, metadata)

    def _simulate(
        self,
        name: str,
        run_dir: Path,
        spec_path: Path,
        body_name: str,
        body_assets_dir: str | Path,
        boxmesh_only: bool,
        max_sim_steps: int | None,
    ) -> Any:
        old_cwd = Path.cwd()
        body_assets_dir = Path(body_assets_dir).resolve()
        sys.path.insert(0, str(self.garmentcode_root))
        sys.path.insert(0, str(self.warp_root))
        Path("/tmp/templatepattern-final-cache").mkdir(parents=True, exist_ok=True)
        Path("/tmp/matplotlib-templatepattern-final").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/templatepattern-final-cache")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-templatepattern-final")
        # Must be set before GarmentCode imports pyrender/PyOpenGL.  On a
        # workstation without an X display, PyOpenGL otherwise chooses GLX
        # and fails only after the expensive simulation has completed.
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from .render_support import configure_opengl
        configure_opengl()
        os.chdir(self.garmentcode_root)
        try:
            import pygarment.data_config as data_config
            import pygarment.garmentcode.utils as garment_utils_module
            import warp.collision.panel_assignment as panel_assignment_module
            import pygarment.meshgen.boxmeshgen as boxmeshgen_module
            import pygarment.meshgen.garment as garment_module
            import pygarment.pattern.utils as pattern_utils_module
            import pygarment.pattern.wrappers as pattern_wrappers_module
            import pygarment.meshgen.render.texture_utils as texture_utils_module
            import pygarment.meshgen.sim_config as sim_config_module
            from pygarment.meshgen.boxmeshgen import BoxMesh
            from pygarment.meshgen.sim_config import PathCofig
            from pygarment.meshgen.simulation import run_sim

            self._patch_body_assets(sim_config_module, Path(body_assets_dir).resolve())
            self._patch_numpy2_vector_angles(
                pattern_utils_module,
                pattern_wrappers_module,
                garment_utils_module,
            )
            self._patch_checks(boxmeshgen_module)
            self._patch_uv_compat(boxmeshgen_module, texture_utils_module)
            self._patch_panel_assignment_cpu(panel_assignment_module, garment_module)
            self._patch_edge_contact_budget(garment_module.wp)
            props = data_config.Properties(str(self.sim_config))
            if max_sim_steps is not None:
                props["sim"]["config"]["max_sim_steps"] = max_sim_steps
            paths = PathCofig(
                in_element_path=spec_path.parent,
                out_path=run_dir / "simulation",
                in_name=name,
                out_name=name,
                body_name=body_name,
                smpl_body=True,
                add_timestamp=False,
            )
            self._flatten_paths(paths, run_dir / "simulation", name)
            mesh = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
            mesh.load()
            mesh.serialize(paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"])
            props.serialize(paths.element_sim_props)
            if not boxmesh_only:
                render_error = None
                try:
                    run_sim(mesh.name, props, paths, save_v_norms=False, store_usd=False, optimize_storage=False, verbose=False)
                except BaseException as exc:
                    render_error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    props.serialize(paths.element_sim_props)
                    stats = props['sim']['stats']
                    failures = {key: value for key, value in stats.get('fails', {}).items() if value}
                    write_json(run_dir / 'simulation' / 'simulation_report.json', {
                        'accepted': not failures and render_error is None,
                        'failures': failures, 'stats': stats, 'pipeline_error': render_error,
                    })
                if failures:
                    raise RuntimeError(f"Simulation quality failed: {', '.join(failures)}; see simulation/simulation_report.json")
                required = [paths.g_sim, paths.render_path('front'), paths.render_path('back')]
                missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
                if missing:
                    raise RuntimeError(f"Missing simulation outputs: {missing}")
            return paths
        finally:
            os.chdir(old_cwd)

    def _flatten_paths(self, paths: Any, sim_dir: Path, name: str) -> None:
        old_out_el = Path(paths.out_el)
        sim_dir.mkdir(parents=True, exist_ok=True)
        paths.out = sim_dir
        paths.out_el = sim_dir
        if old_out_el != sim_dir and old_out_el.exists():
            try:
                old_out_el.rmdir()
            except OSError:
                pass
        paths.g_box_mesh = sim_dir / "boxmesh.obj"
        paths.g_box_mesh_compressed = sim_dir / "boxmesh.ply"
        paths.g_mesh_segmentation = sim_dir / "sim_segmentation.txt"
        paths.g_orig_edge_len = sim_dir / "orig_lens.pickle"
        paths.g_vert_labels = sim_dir / "vertex_labels.yaml"
        paths.g_texture_fabric = sim_dir / "texture_fabric.png"
        paths.g_texture = sim_dir / "texture.png"
        paths.g_mtl = sim_dir / "material.mtl"
        paths.g_specs = sim_dir / "garment_specification.json"
        paths.element_sim_props = sim_dir / "sim_props.yaml"
        paths.body_mes = sim_dir / "body_measurements.yaml"
        paths.design_params = sim_dir / "design_params.yaml"
        paths.g_sim = sim_dir / "sim.obj"
        paths.g_sim_glb = sim_dir / "sim.glb"
        paths.g_sim_compressed = sim_dir / "sim.ply"
        paths.usd = sim_dir / "simulation.usd"
        paths.render_path = lambda camera_name="": sim_dir / (f"render_{camera_name}.png" if camera_name else "render.png")

    def _patch_body_assets(self, sim_config_module: Any, body_assets_dir: Path) -> None:
        original = sim_config_module.Properties

        def properties(path: str) -> Any:
            props = original(path)
            if Path(path).name == "system.json":
                props["bodies_default_path"] = str(body_assets_dir)
            return props

        sim_config_module.Properties = properties

    def _patch_checks(self, boxmeshgen_module: Any) -> None:
        def collapse_with_junctions(self: Any) -> None:
            stitched = self._stitch_vertices()
            valid, invalid = self._is_stitching_valid(stitched, front_end_only=True)
            if not valid:
                raise boxmeshgen_module.StitchingError(f"invalid stitching: {invalid[:20]}")

        boxmeshgen_module.BoxMesh.collapse_stitch_vertices = collapse_with_junctions

    def _patch_numpy2_vector_angles(
        self,
        pattern_utils_module: Any,
        pattern_wrappers_module: Any,
        garment_utils_module: Any,
    ) -> None:
        """Keep GarmentCode's signed 2D angle helper working with NumPy 2.x.

        NumPy 2 rejects ``np.cross([x, y], [x, y])``.  GarmentCode uses that
        expression only to obtain the signed scalar z component, so calculate
        that component explicitly while retaining the original 3D behavior.
        """
        import numpy as np

        def vector_angle(v1: Any, v2: Any) -> float:
            first = np.asarray(v1, dtype=float)
            second = np.asarray(v2, dtype=float)
            if first.shape != second.shape or first.ndim != 1 or first.size not in (2, 3):
                raise ValueError(f"vector_angle expects equally shaped 2D or 3D vectors, got {first.shape} and {second.shape}")
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator == 0.0:
                raise ValueError("vector_angle cannot use a zero-length vector")
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            angle = float(np.arccos(cosine))
            cross = float(first[0] * second[1] - first[1] * second[0]) if first.size == 2 else np.cross(first, second)
            if np.any(np.abs(cross) > 1.0e-5):
                angle *= float(np.sign(cross[-1] if np.ndim(cross) else cross))
            return angle

        pattern_utils_module.vector_angle = vector_angle
        # wrappers.py imported vector_angle with ``from .utils import *``.
        pattern_wrappers_module.vector_angle = vector_angle
        garment_utils_module.vector_angle = vector_angle

    def _patch_uv_compat(self, boxmeshgen_module: Any, texture_utils_module: Any) -> None:
        import numpy as np

        def unwrap_components(result: Any) -> tuple[Any, int]:
            if isinstance(result, tuple):
                if len(result) >= 2 and isinstance(result[0], (int, np.integer)):
                    labels = np.asarray(result[1], dtype=np.int64)
                    count = int(result[0])
                    return labels, count
                labels = np.asarray(result[0], dtype=np.int64)
            else:
                labels = np.asarray(result, dtype=np.int64)
            count = int(labels.max()) + 1 if labels.size else 0
            return labels, count

        def uv_connected_components(face_texture_coords: Any) -> tuple[Any, Any, int]:
            face_components, num_ccs = unwrap_components(texture_utils_module.igl.facet_components(face_texture_coords))
            vertex_result = texture_utils_module.igl.vertex_components(face_texture_coords)
            vert_components, _ = unwrap_components(vertex_result)
            return vert_components, face_components, num_ccs

        texture_utils_module._uv_connected_components = uv_connected_components

        def save_box_mesh_obj_with_fallback(mesh: Any, with_normals: bool = False, in_uv_config: dict[str, Any] | None = None, mat_name: str = "panels_texture") -> None:
            if not mesh.loaded:
                print(f"{mesh.__class__.__name__}::{mesh.name}::WARNING::Pattern is not yet loaded. Nothing saved")
                return
            uv_config = {
                "seam_width": 0.5,
                "dpi": 600,
                "fabric_grain_texture_path": None,
                "fabric_grain_resolution": 1,
            }
            if in_uv_config:
                uv_config.update(in_uv_config)
            try:
                uvs = boxmeshgen_module.texture_mesh_islands(
                    texture_coords=np.array(mesh.vertex_texture),
                    face_texture_coords=np.array([[tex0, tex1, tex2] for _, tex0, _, tex1, _, tex2 in mesh.faces_with_texture]),
                    out_texture_image_path=mesh.paths.g_texture,
                    out_fabric_tex_image_path=mesh.paths.g_texture_fabric,
                    out_mtl_file_path=mesh.paths.g_mtl,
                    boundary_width=uv_config["seam_width"],
                    dpi=uv_config["dpi"],
                    background_img_path=uv_config["fabric_grain_texture_path"],
                    background_resolution=uv_config["fabric_grain_resolution"],
                    mat_name=mat_name,
                )
                mtl_file_name = mesh.paths.g_mtl.name
            except Exception as exc:
                print(f"{mesh.__class__.__name__}::{mesh.name}::WARNING::UV export failed, using normalized fallback UVs: {exc}")
                uvs, _width = texture_utils_module.normalize_UVs(np.array(mesh.vertex_texture), axis_padding=3)
                mtl_file_name = None
                mat_name = None
            boxmeshgen_module.save_obj(
                mesh.paths.g_box_mesh,
                mesh.vertices,
                mesh.faces_with_texture,
                uvs,
                vert_normals=mesh.eval_vertex_normals() if with_normals else None,
                mtl_file_name=mtl_file_name,
                mat_name=mat_name,
            )

        boxmeshgen_module.BoxMesh.save_box_mesh_obj = save_box_mesh_obj_with_fallback

    def _patch_panel_assignment_cpu(self, panel_assignment_module: Any, garment_module: Any) -> None:
        original = panel_assignment_module.panel_assignment
        if getattr(original, "_templatepattern_force_cpu", False):
            return

        def panel_assignment_cpu(*args: Any, **kwargs: Any) -> Any:
            kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        panel_assignment_cpu._templatepattern_force_cpu = True  # type: ignore[attr-defined]
        panel_assignment_module.panel_assignment = panel_assignment_cpu
        garment_module.assign.panel_assignment = panel_assignment_cpu

    def _patch_edge_contact_budget(self, wp_module: Any) -> None:
        model_builder = wp_module.sim.ModelBuilder
        original = model_builder.__init__
        if getattr(original, "_templatepattern_edge_contact_budget", False):
            return

        def init_with_larger_edge_budget(self: Any, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.edge_contact_max = max(int(self.edge_contact_max), 3 * 256)

        init_with_larger_edge_budget._templatepattern_edge_contact_budget = True  # type: ignore[attr-defined]
        model_builder.__init__ = init_with_larger_edge_budget

    def _manifest(
        self,
        name: str,
        run_dir: Path,
        spec_path: Path,
        body_name: str,
        body_assets_dir: str | Path,
        paths: Any,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        def existing(path: Any) -> str | None:
            return str(path) if path is not None and Path(path).exists() else None

        outputs = {
            "run_dir": str(run_dir),
            "boxmesh_obj": None if paths is None else existing(paths.g_box_mesh),
            "sim_obj": None if paths is None else existing(paths.g_sim),
            "sim_glb": None if paths is None else existing(paths.g_sim_glb),
            "render_front": None if paths is None else existing(paths.render_path("front")),
            "render_back": None if paths is None else existing(paths.render_path("back")),
        }
        manifest = {
            "pipeline": "final_garmentcode_boxmesh_sim",
            "name": name,
            "garmentcode_spec": str(spec_path),
            "boxmesh_obj": outputs["boxmesh_obj"],
            "sim_obj": outputs["sim_obj"],
            "render_front": outputs["render_front"],
            "render_back": outputs["render_back"],
            "sim_config": str(self.sim_config),
            "body_name": body_name,
            "smpl_body": True,
            "body_assets_dir": str(body_assets_dir),
            "outputs": outputs,
        }
        if metadata:
            manifest.update(metadata)
        return manifest
