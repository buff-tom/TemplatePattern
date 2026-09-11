"""Serial GPU acceptance run with persisted logs and explicit failure reports."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from TemplatePattern_final.sim.stage2_pipeline import SimulationStage2Pipeline
from TemplatePattern_final.shared.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-report-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smpl-model-dir', type=Path, required=True)
    parser.add_argument('--max-sim-steps', type=int)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(f'Use a new output directory: {root}')
    root.mkdir(parents=True)
    cases = []
    for style in ('short', 'long'):
        item = {'style': style, 'directory': style}
        with (root / f'{style}.log').open('w') as log:
            try:
                with redirect_stdout(log), redirect_stderr(log):
                    pipeline = SimulationStage2Pipeline(
                        stage1_dir=args.height_report_dir / f'{style}_1800_proportional/stage1',
                        output_dir=root/style, model_dir=args.smpl_model_dir, pose='a30')
                    pipeline.run(max_sim_steps=args.max_sim_steps, archive=True)
                item['status'] = 'completed'
            except Exception as exc:
                item.update(status='failed', error=str(exc))
        cases.append(item)
        write_json(root/'gpu_validation_report.json', {'cases': cases})
        print(item, flush=True)
    if any(item['status'] != 'completed' for item in cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
