from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

from TemplatePattern_final.pattern2d.long_pipeline import LongSleevePatternPipeline
from TemplatePattern_final.pattern2d.short_pipeline import ShortSleevePatternPipeline
from TemplatePattern_final.shared.io import ROOT
from TemplatePattern_final.shared.task_bundle import Stage1Bundle
from TemplatePattern_final.sim.body_fit import BodyFitter, MEASUREMENT_KEYS, TOLERANCES_CM
from TemplatePattern_final.sim.smpl_model import MALE_MODEL_FILENAME, SmplModel
from TemplatePattern_final.sim.spec_converter import GarmentSpecConverter
from TemplatePattern_final.sim.stage2_pipeline import SimulationStage2Pipeline


class StageProtocolTest(unittest.TestCase):
    def test_sleeve_seam_anatomical_directions(self):
        from TemplatePattern_final.sim.piece_splitter import PieceSplitter
        from TemplatePattern_final.sim.curve_ops import semantic_polyline
        with tempfile.TemporaryDirectory() as temporary:
            for style, pipeline in (('short', ShortSleevePatternPipeline), ('long', LongSleevePatternPipeline)):
                root = Path(temporary)/style
                pipeline(ROOT/'config'/f'body_{style}.json', root).run()
                pattern = PieceSplitter().expand(json.loads((root/'pattern.json').read_text()))
                for seam in pattern['seams']:
                    if seam['a']['edge'] not in ('sleeve_cap_front', 'sleeve_cap_yoke', 'underarm_front', 'shoulder'):
                        continue
                    lines = [semantic_polyline(pattern['pieces'][seam[s]['piece']], seam[s]['edge']) for s in ('a', 'b')]
                    delta = [line[-1][1]-line[0][1] for line in lines]
                    self.assertEqual(seam['direction'], 'same' if delta[0]*delta[1] > 0 else 'opposite')

    def test_drawing_y_is_converted_to_world_up(self):
        converter = GarmentSpecConverter()
        self.assertGreater(converter._local_point([0, 0], 0, 10)[1], converter._local_point([0, 20], 0, 10)[1])

    def test_invalid_body_gets_failure_report(self):
        from TemplatePattern_final.cli import run_stage1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((ROOT/'config/body_long.json').read_text())
            config['body_input']['height'] = 1500
            config['body_input']['chest'] = -1
            (root/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, 'positive finite'):
                run_stage1('long_sleeve', root/'config.json', root/'stage1')
            self.assertEqual(json.loads((root/'stage1/stage1_manifest.json').read_text())['status'], 'failed')

    def test_failure_manifest_and_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = SimulationStage2Pipeline(style='long_sleeve', output_dir=temporary)
            with patch.object(pipeline, '_run', side_effect=RuntimeError('static_equilibrium')):
                with self.assertRaisesRegex(RuntimeError, 'failure manifest'):
                    pipeline.run(convert_only=True, archive=True)
            manifest = json.loads((Path(temporary)/'stage2_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('static_equilibrium', manifest['error'])
            with zipfile.ZipFile(Path(temporary)/'stage2_results.zip') as archive:
                self.assertIn('stage2_manifest.json', archive.namelist())

    def test_scaled_semantic_points_share_outline_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = LongSleevePatternPipeline(ROOT/'config/body_long.json', Path(temporary))
            source = {'id': 'front', 'boundary': [[0,0],[10,0],[10,20],[0,20]],
                      'control_points': {'corner': [0,0]},
                      'edges': [{'semantic_role': 'side', 'polyline': [[0,0],[0,20]]}]}
            with patch.object(pipeline, 'piece_scale', return_value=(1.5,2.)):
                result = pipeline.scale_piece(source)
            self.assertEqual(result['boundary'][0], result['points']['corner'])
            self.assertEqual(result['boundary'][0], result['curves']['side'][0]['polyline'][0])

    def test_relative_stage2_output_and_archive(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            relative = Path(temporary).relative_to(Path.cwd()) / "stage2"
            pipeline = SimulationStage2Pipeline(style="long_sleeve", output_dir=relative)
            self.assertEqual(pipeline.run_dir, Path.cwd() / relative)
            body = pipeline.run_dir / "body"
            body.mkdir(parents=True)
            (body / "smpl_vert_segmentation.json").write_text("{}")
            from TemplatePattern_final.shared.io import read_json
            self.assertEqual(read_json(body / "smpl_vert_segmentation.json"), {})
            (pipeline.run_dir / "stage2_manifest.json").write_text("{}")
            for _ in range(2):
                with zipfile.ZipFile(pipeline.create_archive()) as archive:
                    self.assertEqual(set(archive.namelist()), {
                        "body/smpl_vert_segmentation.json", "stage2_manifest.json"
                    })

    def test_stage1_bundles_are_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for style, pipeline, config in (
                ("long_sleeve", LongSleevePatternPipeline, ROOT / "config" / "body_long.json"),
                ("short_sleeve", ShortSleevePatternPipeline, ROOT / "config" / "body_short.json"),
            ):
                original = root / style
                pipeline(config, original).run()
                moved = root / f"moved_{style}"
                shutil.move(original, moved)
                bundle = Stage1Bundle.load(moved)
                self.assertEqual(bundle.style, style)
                self.assertTrue(bundle.pattern.is_file())
                self.assertTrue(bundle.body_target.is_file())
                self.assertFalse(json.loads(bundle.pattern.read_text())["source_template"].startswith("/"))

    def test_dynamic_geometry_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary) / "stage1"
            LongSleevePatternPipeline(ROOT / "config" / "body_long.json", task).run()
            output = Path(temporary) / "garment_specification.json"
            result = GarmentSpecConverter().convert_file(
                task / "pattern.json",
                output,
                "long_sleeve_a30",
            )
            debug = json.loads(Path(result["debug"]).read_text())
            self.assertEqual(debug['geometry_source'], 'stage1_pattern')
            self.assertEqual(debug['skipped_stitches'], [])
            source = json.loads((task / 'pattern.json').read_text())
            converter = GarmentSpecConverter()
            first, _ = converter.convert(source, 'long_sleeve_a30')
            for piece in source['pieces'].values():
                for point in piece.get('boundary', []):
                    point[0] *= 1.1
                for entries in piece.get('curves', {}).values():
                    for entry in entries:
                        for point in entry.get('polyline', []):
                            point[0] *= 1.1
                for point in piece.get('points', {}).values():
                    point[0] *= 1.1
            second, _ = converter.convert(source, 'long_sleeve_a30')
            self.assertNotEqual(first['pattern']['panels'], second['pattern']['panels'])


class DynamicSmplTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TEMPLATEPATTERN_SMPL_MODEL_DIR"), "SMPL model directory is not configured")
    def test_reference_sizes_meet_tolerances(self) -> None:
        model_dir = Path(os.environ["TEMPLATEPATTERN_SMPL_MODEL_DIR"])
        self.assertTrue((model_dir / MALE_MODEL_FILENAME).is_file())
        model = SmplModel(model_dir)
        segmentation = json.loads((ROOT / "assets" / "body_long" / "smpl_vert_segmentation.json").read_text())
        params = json.loads((ROOT / "assets" / "body_long" / "torso_v6_male_asia_M_a45_smpl_params.json").read_text())
        references = json.loads((ROOT / "config" / "size_body_reference_v4.json").read_text())["sizes"]
        fitter = BodyFitter(model, segmentation)
        previous_betas = None
        previous_vertices = None
        previous_correction = None
        for size in ("XS", "L", "XXXL"):
            target = {key: float(references[size][key]) / 10.0 for key in MEASUREMENT_KEYS}
            result = fitter.fit(params, target, max_nfev=60)
            self.assertTrue(result.report["accepted"])
            for key, error in result.report["absolute_errors_cm"].items():
                self.assertLessEqual(error, TOLERANCES_CM[key])
            if previous_betas is not None:
                self.assertNotEqual(previous_betas, result.params["betas"])
                self.assertNotEqual(previous_correction, result.correction)
            vertices = model.canonical_vertices(result.params["betas"], result.correction, segmentation)
            self.assertEqual(len(vertices), 6890)
            self.assertTrue(np.isfinite(vertices).all())
            if previous_vertices is not None:
                self.assertFalse(np.allclose(previous_vertices, vertices))
            from TemplatePattern_final.sim.body_assets import BodyAssetManager
            posed = []
            for angle in (0, 30, 45, 60):
                pose_params = dict(result.params)
                pose_params['body_pose'] = [0.]*69 if angle == 0 else BodyAssetManager()._a_pose_body_pose(None, angle)
                mesh = model.posed_vertices(pose_params, result.correction, segmentation)
                self.assertEqual(mesh.shape, vertices.shape)
                self.assertTrue(np.isfinite(mesh).all())
                posed.append(mesh)
            for a, b in zip(posed, posed[1:]):
                self.assertFalse(np.allclose(a, b))
            previous_betas = result.params["betas"]
            previous_vertices = vertices
            previous_correction = result.correction


if __name__ == "__main__":
    unittest.main()
