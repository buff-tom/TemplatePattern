from __future__ import annotations

import argparse
import json

from TemplatePattern_final.sim.stage2_pipeline import SimulationStage2Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final short-sleeve 3D simulation stage.")
    parser.add_argument("--pattern-json", default=None)
    parser.add_argument("--stage1-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--garmentcode-root", default=None)
    parser.add_argument("--warp-root", default=None)
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--arm-angle-deg", type=float, default=None)
    parser.add_argument("--body-config", default=None)
    parser.add_argument("--body-params", default=None)
    parser.add_argument("--body-target", default=None)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument("--pose", choices=("a30", "a60", "a45", "tpose"), default="a30")
    parser.add_argument("--preserve-old-output", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--boxmesh-only", action="store_true")
    parser.add_argument("--max-sim-steps", type=int, default=None)
    args = parser.parse_args()
    manifest = SimulationStage2Pipeline(
        "short_sleeve",
        args.pattern_json,
        args.output_dir,
        args.garmentcode_root,
        args.warp_root,
        args.sim_config,
        stage1_dir=args.stage1_dir,
        model_dir=args.smpl_model_dir,
        arm_angle_deg=args.arm_angle_deg,
        body_config=args.body_config,
        body_params=args.body_params,
        body_target=args.body_target,
        pose=args.pose,
        preserve_old_output=args.preserve_old_output,
    ).run(convert_only=args.convert_only, boxmesh_only=args.boxmesh_only, max_sim_steps=args.max_sim_steps)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
