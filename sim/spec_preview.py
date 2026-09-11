from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from TemplatePattern_final.shared.geometry import bbox
from TemplatePattern_final.shared.io import read_json


def write_spec_preview(spec_path: str | Path, output_path: str | Path) -> Path:
    spec = read_json(spec_path)
    polygons = []
    for name, panel in spec.get("pattern", {}).get("panels", {}).items():
        points = [_world_point(point, panel.get("translation", [0.0, 0.0, 0.0]), panel.get("rotation", [0.0, 0.0, 0.0])) for point in panel.get("vertices", [])]
        if points:
            polygons.append({"name": name, "points": points})
    svg = _svg(polygons)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def _world_point(point: list[float], translation: list[float], rotation: list[float]) -> list[float]:
    angle = math.radians(float(rotation[2]) if len(rotation) >= 3 else 0.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = float(point[0]) * cos_a - float(point[1]) * sin_a + float(translation[0])
    y = float(point[0]) * sin_a + float(point[1]) * cos_a + float(translation[1])
    return [x, y]


def _svg(polygons: list[dict[str, Any]]) -> str:
    all_points = [point for item in polygons for point in item["points"]]
    if not all_points:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    box = bbox(all_points)
    margin = 12.0
    width = max(1.0, box[2] - box[0] + margin * 2.0)
    height = max(1.0, box[3] - box[1] + margin * 2.0)
    tx = -box[0] + margin
    ty = -box[1] + margin
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.6f} {height:.6f}">',
        '<g fill="#e2a9b5" fill-opacity="0.95" stroke="#6b4f57" stroke-width="0.8">',
    ]
    for item in polygons:
        points = " ".join(f"{point[0] + tx:.6f},{height - (point[1] + ty):.6f}" for point in item["points"])
        lines.append(f'<polygon points="{points}" />')
    lines.append("</g></svg>")
    return "\n".join(lines) + "\n"
