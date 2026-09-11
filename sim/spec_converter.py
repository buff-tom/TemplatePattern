from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from TemplatePattern_final.shared.geometry import bbox, center_box, sequential_edges
from TemplatePattern_final.shared.io import resolve, read_json, write_json

from .boundary_sampler import sample_boundary
from .curve_ops import clean_polyline, same_point, semantic_polyline
from .piece_splitter import PieceSplitter
from .stitch_builder import StitchBuilder
from .dynamic_stitches import build_dynamic_stitches


PROPERTIES = {
    "curvature_coords": "relative",
    "normalize_panel_translation": False,
    "normalized_edge_loops": True,
    "units_in_meter": 100,
}

ROTATION_BY_REGION = {
    "front_torso": [0.0, 0.0, 0.0],
    "back_torso": [0.0, 180.0, 0.0],
    "upper_back": [0.0, 180.0, 0.0],
    "neck": [0.0, 0.0, 0.0],
    "arm_pair": [0.0, 0.0, 90.0],
    "wrist_pair": [0.0, 0.0, 90.0],
}


class GarmentSpecConverter:
    def __init__(self, placement_rules: str | Path = "config/placement_3d_rules_v1.json") -> None:
        placement_path = resolve(placement_rules)
        self.placement = read_json(placement_path) if placement_path.exists() else {"pieces": {}}
        self.stitches = StitchBuilder()

    def convert_file(self, pattern_json: str | Path, output: str | Path, name: str | None = None) -> dict[str, Any]:
        pattern = read_json(pattern_json)
        spec, debug = self.convert(pattern, name or Path(pattern_json).stem)
        write_json(output, spec)
        debug_path = Path(output).parent / "conversion_debug.json"
        write_json(debug_path, debug)
        return {"spec": str(output), "debug": str(debug_path)}

    def convert(self, pattern: dict[str, Any], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        pattern = PieceSplitter().expand(pattern)
        panels: dict[str, Any] = {}
        edge_lookup: dict[tuple[str, str], list[int]] = {}
        for piece_name, piece in pattern.get("pieces", {}).items():
            local_piece = self._local_piece(piece)
            rule = pattern.get("placement_rules", {}).get(piece_name, {})
            panels[piece_name] = self._panel(piece_name, rule, local_piece["boundary"])
            edge_lookup.update(self.stitches.lookup_for_piece(piece_name, local_piece, panels[piece_name]["vertices"]))
        stitches = build_dynamic_stitches(pattern, panels, edge_lookup, self.stitches)
        skipped = []
        spec = {"pattern": {"panels": panels, "stitches": stitches}, "parameters": {}, "parameter_order": [], "properties": dict(PROPERTIES)}
        debug = {"name": name, "style": pattern.get("style"), "panel_count": len(panels), "stitch_count": len(stitches), "skipped_stitches": skipped}
        debug["geometry_source"] = "stage1_pattern"
        debug["seam_mapping"] = "semantic_paired_arclength_resampling_5mm"
        return spec, debug

    def _local_piece(self, piece: dict[str, Any]) -> dict[str, Any]:
        boundary = self._sampled_boundary(piece)
        if len(boundary) < 3:
            boundary = clean_polyline(piece.get("boundary", []))
        cx, cy = center_box(bbox(boundary))
        local_boundary = [self._local_point(p, cx, cy) for p in boundary]
        clean_boundary = self._ensure_ccw(clean_polyline(local_boundary))
        if len(clean_boundary) < 3:
            clean_boundary = self._ensure_ccw(self._remove_degenerate(local_boundary))
        local = {"boundary": clean_boundary, "curves": {}, "points": {}}
        for role, entries in piece.get("curves", {}).items():
            local["curves"][role] = []
            for entry in entries:
                item = dict(entry)
                item["polyline"] = [self._local_point(p, cx, cy) for p in clean_polyline(entry.get("polyline", []))]
                local["curves"][role].append(item)
        for name, point in piece.get("points", {}).items():
            local["points"][name] = self._local_point(point, cx, cy)
        return local

    def _sampled_boundary(self, piece: dict[str, Any]) -> list[list[float]]:
        sampled = sample_boundary(piece)
        if len(sampled) >= 3:
            return clean_polyline(sampled)
        boundary: list[list[float]] = []
        for _order, role, polyline in self._ordered_polylines(piece):
            if len(polyline) < 2:
                continue
            if boundary and not same_point(boundary[-1], polyline[0]) and same_point(boundary[-1], polyline[-1]):
                polyline = list(reversed(polyline))
            if boundary and same_point(boundary[-1], polyline[0]):
                boundary.extend(polyline[1:])
            else:
                boundary.extend(polyline)
        return clean_polyline(boundary)

    def _ordered_polylines(self, piece: dict[str, Any]) -> list[tuple[int, str, list[list[float]]]]:
        if piece.get("preserve_boundary_order"):
            ordered = []
            for index, role in enumerate(piece.get("boundary_order", [])):
                polyline = semantic_polyline(piece, role)
                if len(polyline) >= 2:
                    ordered.append((index, role, polyline))
            return ordered
        source_boundary = clean_polyline(piece.get("boundary", []))
        roles = list(piece.get("boundary_order", [])) or list(piece.get("curves", {}))
        items: list[tuple[int, str, list[list[float]]]] = []
        for fallback_order, role in enumerate(roles):
            polyline = semantic_polyline(piece, role)
            if len(polyline) < 2:
                continue
            start = self._boundary_index(source_boundary, polyline[0])
            end = self._boundary_index(source_boundary, polyline[-1])
            if start is None or end is None or not source_boundary:
                items.append((len(source_boundary) + fallback_order, role, polyline))
                continue
            forward = (end - start) % len(source_boundary)
            backward = (start - end) % len(source_boundary)
            if backward < forward:
                polyline = list(reversed(polyline))
                start = end
            items.append((start, role, polyline))
        return sorted(items, key=lambda item: item[0])

    def _boundary_index(self, boundary: list[list[float]], point: list[float]) -> int | None:
        if not boundary:
            return None
        best = min(range(len(boundary)), key=lambda i: abs(boundary[i][0] - point[0]) + abs(boundary[i][1] - point[1]))
        return best

    def _local_point(self, point: list[float], cx: float, cy: float) -> list[float]:
        # Stage1 uses drawing coordinates (Y down); GarmentCode uses Y up.
        return [round((point[0] - cx) / 10.0, 6), round((cy - point[1]) / 10.0, 6)]

    def _remove_degenerate(self, boundary: list[list[float]]) -> list[list[float]]:
        clean = clean_polyline(boundary)
        unique: list[list[float]] = []
        seen: set[tuple[float, float]] = set()
        for point in clean:
            key = (round(point[0], 6), round(point[1], 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        clean = unique
        changed = True
        while changed and len(clean) > 3:
            changed = False
            kept: list[list[float]] = []
            for index, point in enumerate(clean):
                a = clean[index - 1]
                c = clean[(index + 1) % len(clean)]
                area = abs((point[0] - a[0]) * (c[1] - a[1]) - (point[1] - a[1]) * (c[0] - a[0]))
                span = abs(c[0] - a[0]) + abs(c[1] - a[1])
                if span > 1.0e-8 and area / span < 1.0e-5:
                    changed = True
                    continue
                kept.append(point)
            clean = kept
        return clean

    def _ensure_ccw(self, boundary: list[list[float]]) -> list[list[float]]:
        area = 0.0
        for index, point in enumerate(boundary):
            nxt = boundary[(index + 1) % len(boundary)]
            area += point[0] * nxt[1] - nxt[0] * point[1]
        return list(reversed(boundary)) if area < 0 else boundary

    def _panel(self, piece_name: str, rule: dict[str, Any], vertices: list[list[float]]) -> dict[str, Any]:
        rule = rule or self.placement.get("pieces", {}).get(piece_name, {})
        translation, rotation = self._pose(piece_name, rule)
        return {"translation": translation, "rotation": rotation, "vertices": vertices, "edges": sequential_edges(len(vertices)), "label": self._label(piece_name)}

    def _pose(self, piece_name: str, rule: dict[str, Any]) -> tuple[list[float], list[float]]:
        region = rule.get("region", self._region(piece_name))
        side = rule.get("side")
        x = -13.5 if side == "right" else 13.5 if side == "left" else 0.0
        y = 110.0 if "body" in piece_name else 140.0 if "yoke" in piece_name else 155.0
        z = float(rule.get("offset", 30.0))
        if "back" in piece_name and "sleeve" not in piece_name:
            z = -z
        return [x, y, z], ROTATION_BY_REGION.get(region, [0.0, 0.0, 0.0])

    def _region(self, piece_name: str) -> str:
        if "sleeve" in piece_name:
            return "arm_pair"
        if "cuff" in piece_name:
            return "wrist_pair"
        if "collar" in piece_name:
            return "neck"
        if "back" in piece_name:
            return "back_torso"
        return "front_torso"

    def _label(self, piece_name: str) -> str:
        if piece_name.endswith("_right") and ("sleeve" in piece_name or "cuff" in piece_name):
            return "right_arm"
        if "sleeve" in piece_name or "cuff" in piece_name:
            return "left_arm"
        return "body"

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert final pattern JSON to GarmentCode specification.")
    parser.add_argument("pattern_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    result = GarmentSpecConverter().convert_file(args.pattern_json, args.output, args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
