from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import bbox
from .io import write_json


SCHEMA_VERSION = 1
FIT_TOLERANCES_CM = {
    "height": 1.5,
    "chest": 2.0,
    "waist": 2.0,
    "hip": 2.0,
    "shoulder_width": 1.5,
    "arm_length": 1.5,
}


def write_pattern_outputs(out_dir: Path, pattern: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern_path = out_dir / "pattern.json"
    svg_path = out_dir / "pattern.svg"
    body_target_path = out_dir / "body_target.json"
    request_path = out_dir / "stage2_request.json"
    manifest_path = out_dir / "stage1_manifest.json"
    write_json(pattern_path, pattern)
    svg_path.write_text(pattern_svg(pattern), encoding="utf-8")
    body_target = {
        "schema_version": SCHEMA_VERSION,
        "sample_name": manifest["sample_name"],
        "gender": "male",
        "unit": "mm",
        # body_input is deliberately the resolved Stage2 target.  Preserve the
        # user's original request separately when fallback selected a bounded
        # reference capacity.
        "body_input": manifest["body_input"],
        "effective_body_input": manifest.get("effective_body_input", manifest["body_input"]),
        "requested_body_input": manifest.get("requested_body_input", manifest["body_input"]),
        "input_resolution": manifest.get("input_resolution", {"mode": "requested_input", "adjusted": False}),
        "fit_tolerances_cm": FIT_TOLERANCES_CM,
    }
    request = {
        "schema_version": SCHEMA_VERSION,
        "style": pattern["style"],
        "pattern": "pattern.json",
        "body_target": "body_target.json",
        "default_pose": "a30",
        "supported_poses": ["a30", "a45", "a60", "tpose"],
        "body_model": "mhr",
        "sim_config": "default_sim_props.yaml",
    }
    write_json(body_target_path, body_target)
    write_json(request_path, request)
    payload = dict(manifest)
    payload.update({"schema_version": SCHEMA_VERSION, "status": "completed", "stage": "stage1"})
    payload.update({
        "body_model": request["body_model"],
        "default_pose": request["default_pose"],
        "supported_poses": request["supported_poses"],
    })
    payload["outputs"] = {
        "pattern": "pattern.json",
        "svg": "pattern.svg",
        "body_target": "body_target.json",
        "stage2_request": "stage2_request.json",
    }
    write_json(manifest_path, payload)
    return {
        "stage1_dir": str(out_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "pattern": str(pattern_path.resolve()),
        "svg": str(svg_path.resolve()),
        "body_target": str(body_target_path.resolve()),
        "stage2_request": str(request_path.resolve()),
    }


def pattern_svg(pattern: dict[str, Any]) -> str:
    pieces = pattern.get("pieces", {})
    packed = _pack_pieces(pieces)
    all_points = [pt for item in packed for pt in item["points"]]
    if not all_points:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    box = bbox(all_points)
    margin = 24.0
    width = max(box[2] - box[0] + margin * 2, 1.0)
    height = max(box[3] - box[1] + margin * 2, 1.0)
    tx = -box[0] + margin
    ty = -box[1] + margin
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}">',
        '<g fill="none" stroke="#111827" stroke-width="1.2">',
    ]
    for item in packed:
        pts = " ".join(f"{p[0] + tx:.3f},{p[1] + ty:.3f}" for p in item["points"])
        if pts:
            lines.append(f'<polygon points="{pts}" />')
            label = bbox(item["points"])
            lines.append(f'<text x="{label[0] + tx:.3f}" y="{label[1] + ty - 6:.3f}" font-size="14" fill="#374151">{item["name"]}</text>')
    lines.append("</g></svg>")
    return "\n".join(lines) + "\n"


def _pack_pieces(pieces: dict[str, Any]) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    row_x = 0.0
    row_y = 0.0
    row_height = 0.0
    gap = 56.0
    max_row_width = 1100.0
    for name, piece in pieces.items():
        boundary = piece.get("boundary", [])
        if not boundary:
            continue
        box = bbox(boundary)
        width = box[2] - box[0]
        height = box[3] - box[1]
        if row_x and row_x + width > max_row_width:
            row_x = 0.0
            row_y += row_height + gap
            row_height = 0.0
        dx = row_x - box[0]
        dy = row_y - box[1]
        points = [[point[0] + dx, point[1] + dy] for point in boundary]
        layout.append({"name": name, "points": points})
        row_x += width + gap
        row_height = max(row_height, height)
    return layout
