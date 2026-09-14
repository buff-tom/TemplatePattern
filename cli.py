from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from contextlib import redirect_stdout
from typing import Any

from TemplatePattern_final.long_sleeve.run_stage1_pattern import LongSleevePatternPipeline
from TemplatePattern_final.short_sleeve.run_stage1_pattern import ShortSleevePatternPipeline
from TemplatePattern_final.shared.io import read_json, write_json
from TemplatePattern_final.shared.task_bundle import Stage1Bundle
from TemplatePattern_final.sim.stage2_pipeline import SimulationStage2Pipeline


PIPELINES = {"long_sleeve": LongSleevePatternPipeline, "short_sleeve": ShortSleevePatternPipeline}
POSES = ("a30", "a45", "a60", "tpose")
JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="templatepattern", description="Two-stage body-to-pattern and GarmentCode simulation pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    stage1 = sub.add_parser("stage1", help="Create a portable Stage1 task bundle.")
    _stage1_arguments(stage1)

    stage2 = sub.add_parser("stage2", help="Run Stage2 from one Stage1 task bundle.")
    _stage2_arguments(stage2)

    batch1 = sub.add_parser("batch-stage1", help="Create Stage1 bundles for every JSON body config in a directory.")
    batch1.add_argument("--input-dir", type=Path, required=True)
    batch1.add_argument("--output-root", type=Path, required=True)
    batch1.add_argument("--style", choices=tuple(PIPELINES), required=True)
    batch1.add_argument("--resume", action="store_true")
    batch1.add_argument("--fail-fast", action="store_true")

    batch2 = sub.add_parser("batch-stage2", help="Run Stage2 for jobs declared by a Stage1 batch manifest.")
    batch2.add_argument("--stage1-root", type=Path, required=True)
    batch2.add_argument("--output-root", type=Path, required=True)
    batch2.add_argument("--pose", choices=POSES, default="a30")
    batch2.add_argument("--resume", action="store_true")
    batch2.add_argument("--fail-fast", action="store_true")
    _engine_arguments(batch2)
    return parser


def _stage1_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--style", choices=tuple(PIPELINES), required=True)
    parser.add_argument("--body-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _stage2_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pose", choices=POSES, default=None)
    _engine_arguments(parser)


def _engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smpl-model-dir", type=Path, default=None)
    parser.add_argument("--garmentcode-root", type=Path, default=None)
    parser.add_argument("--warp-root", type=Path, default=None)
    parser.add_argument("--sim-config", type=Path, default=None)
    parser.add_argument("--arm-angle-deg", type=float, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--convert-only", action="store_true")
    modes.add_argument("--boxmesh-only", action="store_true")
    parser.add_argument("--max-sim-steps", type=int, default=None)
    parser.add_argument("--archive", action="store_true", help="Write stage2_results.zip inside the output directory.")
    parser.add_argument('--draco', action=argparse.BooleanOptionalAction, default=None, help='Full simulation exports scene_draco.glb by default; --no-draco disables it. Decoder path /draco/.')


def run_stage1(style: str, body_config: Path, output_dir: Path) -> dict[str, Any]:
    body_config = body_config.resolve()
    output_dir = output_dir.resolve()
    _require_persistent_output(output_dir, "/data/stage1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage1 output directory is not empty: {output_dir}")
    try:
        outputs = PIPELINES[style](body_config, output_dir).run()
    except Exception as exc:
        write_json(output_dir / 'stage1_manifest.json', {
            'schema_version': 1, 'stage': 'stage1', 'status': 'failed',
            'style': style, 'error': f'{type(exc).__name__}: {exc}',
        })
        raise RuntimeError(f"{exc}; failure manifest: {output_dir / 'stage1_manifest.json'}") from exc
    return {"status": "completed", "stage": "stage1", **outputs}


def run_stage2(args: argparse.Namespace, stage1_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    bundle = Stage1Bundle.load(stage1_dir or args.stage1_dir)
    target = output_dir or args.output_dir
    _require_persistent_output(target, "/data/stage2")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Stage2 output directory is not empty: {target}")
    pose = args.pose or str(bundle.request.get("default_pose") or "a30")
    _require_gpu_if_needed(args.convert_only, args.boxmesh_only)
    return SimulationStage2Pipeline(
        stage1_dir=bundle.root,
        output_dir=target,
        garmentcode_root=args.garmentcode_root,
        warp_root=args.warp_root,
        sim_config=args.sim_config,
        model_dir=args.smpl_model_dir,
        arm_angle_deg=args.arm_angle_deg,
        pose=pose,
    ).run(convert_only=args.convert_only, boxmesh_only=args.boxmesh_only, max_sim_steps=args.max_sim_steps, archive=args.archive, draco=args.draco)


def batch_stage1(args: argparse.Namespace) -> dict[str, Any]:
    _require_persistent_output(args.output_root, "/data/stage1")
    configs = sorted(args.input_dir.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"no JSON body configs found in {args.input_dir}")
    previous = _optional_json(args.output_root / "batch_manifest.json")
    previous_jobs = {job["job_id"]: job for job in previous.get("jobs", [])} if previous else {}
    jobs = []
    for config in configs:
        job_id = _job_id(config.stem)
        digest = _file_digest(config)
        output = args.output_root / job_id
        old = previous_jobs.get(job_id, {})
        if args.resume and old.get("status") == "completed" and old.get("input_digest") == digest:
            try:
                Stage1Bundle.load(output)
                jobs.append(old)
                continue
            except (OSError, ValueError):
                pass
        try:
            if args.resume and output.exists() and any(output.iterdir()):
                _archive_partial(output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"output exists; remove it or use an unchanged completed job with --resume: {output}")
            run_stage1(args.style, config, output)
            jobs.append({"job_id": job_id, "status": "completed", "stage1_dir": job_id, "input_digest": digest})
        except Exception as exc:
            print(f"[{job_id}] Stage1 failed: {exc}", file=sys.stderr)
            jobs.append({"job_id": job_id, "status": "failed", "stage1_dir": job_id, "input_digest": digest, "error": str(exc)})
            if args.fail_fast:
                break
    return _write_batch_manifest(args.output_root, "stage1", jobs)


def batch_stage2(args: argparse.Namespace) -> dict[str, Any]:
    _require_persistent_output(args.output_root, "/data/stage2")
    batch_path = args.stage1_root / "batch_manifest.json"
    batch = read_json(batch_path)
    if batch.get("stage") != "stage1":
        raise ValueError(f"not a Stage1 batch manifest: {batch_path}")
    jobs = []
    for source_job in batch.get("jobs", []):
        job_id = _job_id(str(source_job.get("job_id") or ""))
        if source_job.get("status") != "completed":
            jobs.append({"job_id": job_id, "status": "blocked", "error": "Stage1 did not complete"})
            continue
        stage1_dir = args.stage1_root / source_job["stage1_dir"]
        bundle = Stage1Bundle.load(stage1_dir)
        output = args.output_root / job_id / args.pose
        existing = _optional_json(output / "stage2_manifest.json")
        if args.resume and _stage2_complete(output, existing, bundle.digest()):
            jobs.append({"job_id": job_id, "status": "completed", "stage2_dir": f"{job_id}/{args.pose}", "resumed": True})
            continue
        try:
            if args.resume and output.exists() and any(output.iterdir()):
                _archive_partial(output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Stage2 output exists and is not resumable: {output}")
            run_stage2(args, stage1_dir, output)
            jobs.append({"job_id": job_id, "status": "completed", "stage2_dir": f"{job_id}/{args.pose}"})
        except Exception as exc:
            print(f"[{job_id}] Stage2 failed: {exc}", file=sys.stderr)
            jobs.append({"job_id": job_id, "status": "failed", "stage2_dir": f"{job_id}/{args.pose}", "error": str(exc)})
            if args.fail_fast:
                break
    return _write_batch_manifest(args.output_root, "stage2", jobs, pose=args.pose)


def _write_batch_manifest(root: Path, stage: str, jobs: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    counts = {status: sum(job.get("status") == status for job in jobs) for status in ("completed", "failed", "blocked")}
    payload = {"schema_version": 1, "stage": stage, "status": "completed" if not counts["failed"] else "completed_with_errors", "counts": counts, "jobs": jobs, **extra}
    path = root / "batch_manifest.json"
    write_json(path, payload)
    return {**payload, "manifest": str(path.resolve())}


def _require_persistent_output(path: Path, docker_root: str) -> None:
    if os.environ.get("TEMPLATEPATTERN_IN_DOCKER") != "1":
        return
    root = Path(docker_root)
    if not root.is_mount():
        raise RuntimeError(f"{root} is not a mounted Docker volume; refusing to create ephemeral outputs")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Docker output must be inside {root}: {path}") from exc


def _require_gpu_if_needed(convert_only: bool, boxmesh_only: bool) -> None:
    if os.environ.get("TEMPLATEPATTERN_IN_DOCKER") == "1" and not convert_only and not boxmesh_only:
        if not Path("/dev/nvidiactl").exists() and not list(Path("/dev").glob("nvidia[0-9]*")):
            raise RuntimeError("full simulation requires an NVIDIA GPU; run Docker with --gpus all or use --convert-only")


def _job_id(value: str) -> str:
    if not JOB_ID.fullmatch(value):
        raise ValueError(f"invalid job ID {value!r}; use letters, digits, dot, underscore, or hyphen")
    return value


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_partial(path: Path) -> Path:
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.partial-{index}")
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        index += 1


def _stage2_complete(root: Path, manifest: dict[str, Any], stage1_digest: str) -> bool:
    if manifest.get("status") != "completed" or manifest.get("stage1_digest") != stage1_digest or manifest.get('stop_after') != 'simulation':
        return False
    required = ('boxmesh', 'simulation', 'render_front', 'render_back')
    if any(not manifest.get('outputs', {}).get(key) for key in required):
        return False
    for value in (manifest.get("outputs") or {}).values():
        if value is None:
            continue
        path = (root / str(value)).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def _optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "stage1":
                result = run_stage1(args.style, args.body_config, args.output_dir)
            elif args.command == "stage2":
                result = run_stage2(args)
            elif args.command == "batch-stage1":
                result = batch_stage1(args)
            else:
                result = batch_stage2(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "completed_with_errors":
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
