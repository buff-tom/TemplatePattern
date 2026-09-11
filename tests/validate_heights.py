"""Reproducible height sweep; records failures as well as successes.

Inputs and reports are written under --output-dir, never into existing tasks.
This is finite coverage, not proof that arbitrary body proportions are feasible.
"""
import argparse
import json
from pathlib import Path

from TemplatePattern_final.shared.io import ROOT, write_json
from TemplatePattern_final.pattern2d.long_pipeline import LongSleevePatternPipeline
from TemplatePattern_final.pattern2d.short_pipeline import ShortSleevePatternPipeline
from TemplatePattern_final.sim.spec_converter import GarmentSpecConverter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smpl-model-dir', type=Path)
    parser.add_argument('--boxmesh', action='store_true', help='Also triangulate six representative cases using local GarmentCode')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    records = []
    for height in range(1500, 2201, 100):
        for mode in ('height_only', 'proportional'):
            for style, pipeline in (('long', LongSleevePatternPipeline), ('short', ShortSleevePatternPipeline)):
                config = json.loads((ROOT / 'config' / f'body_{style}.json').read_text())
                if mode == 'proportional':
                    factor = height / config['body_input']['height']
                    config['body_input'] = {k: v * factor for k, v in config['body_input'].items()}
                config['body_input']['height'] = height
                job = root / f'{style}_{height}_{mode}'
                write_json(job / 'body_config.json', config)
                record = {'height_mm': height, 'mode': mode, 'style': style, 'directory': str(job.relative_to(root))}
                try:
                    pipeline(job / 'body_config.json', job / 'stage1').run()
                    record['stage1'] = 'passed'
                    GarmentSpecConverter().convert_file(job / 'stage1/pattern.json', job / 'conversion/garment_specification.json', f'{style}_sleeve_a30')
                    record['conversion'] = 'passed'
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                records.append(record)
                write_json(root / 'height_validation_report.json', {'cases': records, 'scope': 'Stage1 and specification conversion, not BoxMesh or simulation'})
    if args.smpl_model_dir:
        from TemplatePattern_final.sim.smpl_model import SmplModel
        from TemplatePattern_final.sim.body_fit import BodyFitter, MEASUREMENT_KEYS, BodyFitError
        segmentation = json.loads((ROOT / 'assets/body_long/smpl_vert_segmentation.json').read_text())
        params = json.loads((ROOT / 'assets/body_long/torso_v6_male_asia_M_a45_smpl_params.json').read_text())
        fitter = BodyFitter(SmplModel(args.smpl_model_dir), segmentation)
        base = json.loads((ROOT / 'config/body_long.json').read_text())['body_input']
        bodies = []
        for height in (1500, 1510, 1700, 1900, 2200, 2600):
            for mode in ('height_only', 'proportional'):
                target = {k: base[k] / 10 * (height / base['height'] if mode == 'proportional' else 1) for k in MEASUREMENT_KEYS}
                target['height'] = height / 10
                item = {'height_mm': height, 'mode': mode}
                try:
                    result = fitter.fit(params, target, max_nfev=60)
                    item.update(status='passed', report=result.report, params=result.params)
                except BodyFitError as exc:
                    item.update(status='failed', error=str(exc), report=exc.report)
                except Exception as exc:
                    item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                bodies.append(item)
                write_json(root / 'smpl_height_report.json', {'cases': bodies})
                print(f"SMPL {height} {mode}: {item['status']}", flush=True)
    if args.boxmesh:
        import sys
        from TemplatePattern_final.sim.garmentcode_runner import GarmentCodeRunner
        runner = GarmentCodeRunner()
        sys.path.insert(0, str(runner.garmentcode_root))
        import pygarment.meshgen.boxmeshgen as bm
        runner._patch_checks(bm)
        meshes = []
        for height in (1500, 1800, 2200):
            for style in ('long', 'short'):
                item = {'height_mm': height, 'style': style}
                try:
                    mesh = bm.BoxMesh(root / f'{style}_{height}_proportional/conversion/garment_specification.json', 1.0)
                    mesh.load()
                    item.update(status='passed', vertices=len(mesh.vertices), faces=len(mesh.faces))
                except Exception as exc:
                    item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                meshes.append(item)
                write_json(root / 'boxmesh_height_report.json', {'cases': meshes})
                print(item, flush=True)
    print(root / 'height_validation_report.json')


if __name__ == '__main__':
    main()
