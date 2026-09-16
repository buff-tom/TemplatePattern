from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from TemplatePattern_final.shared.body_config import BodyConfig
from TemplatePattern_final.shared.geometry import bbox, scale_points
from TemplatePattern_final.shared.io import ROOT, read_json, relative_to_root
from TemplatePattern_final.shared.pattern_result import write_pattern_outputs
from TemplatePattern_final.shared.template_selection import select_template

from .schema import BOUNDARY_ORDERS, LONG_SEAMS, PIECE_ORDER, PLACEMENT_RULES, SHORT_SEAMS, SIZE_ORDER


class TemplatePatternPipeline:
    style = ""
    garment_type = ""
    template_dir = Path()

    def __init__(self, body_config: str | Path, output_dir: str | Path | None = None) -> None:
        self.body = BodyConfig.load(body_config)
        self.output_dir = Path(output_dir) if output_dir else ROOT / "outputs" / self.style / "stage1"
        self.references = read_json("config/size_body_reference_v4.json")
        self.size_refs = self.references['sizes']
        self.selection = select_template(self.body.body_input, self.references)
        self.requested_body_input = dict(self.body.body_input)
        fallback = self.selection["selection_strategy"] == "covering_fallback"
        self.effective_body_input = (
            {key: float(value) for key, value in self.selection["selected_capacity_mm"].items()}
            if fallback
            else dict(self.requested_body_input)
        )
        changed = {
            key: {
                "requested_mm": self.requested_body_input[key],
                "effective_mm": self.effective_body_input[key],
            }
            for key in self.requested_body_input
            if abs(self.requested_body_input[key] - self.effective_body_input[key]) > 1.0e-9
        }
        self.input_resolution = {
            "mode": "selected_capacity_fallback" if fallback else "requested_input",
            "adjusted": bool(changed),
            "selected_anchor": self.selection["nearest_anchor"],
            "selected_size": self.selection["nearest_size"],
            "changed_measurements": changed,
        }

    def run(self) -> dict[str, str]:
        size = self.select_size()
        template_path = self.template_path(size)
        template = read_json(template_path)
        pattern = self.convert_template(template, template_path, size)
        manifest = {
            "status": "generated",
            "variant": f"final_{self.style}",
            "sample_name": self.body.sample_name,
            "unit": "mm",
            # body_input is the Stage2 contract and therefore always contains
            # the measurements actually used to generate both pattern/body.
            "body_input": self.effective_body_input,
            "effective_body_input": self.effective_body_input,
            "requested_body_input": self.requested_body_input,
            "input_resolution": self.input_resolution,
            "nearest_template_selection": {**self.selection, "selection_source": "config/size_body_reference_v4.json"},
        }
        return write_pattern_outputs(self.output_dir, pattern, manifest)

    def select_size(self) -> str:
        return self.selection['nearest_size']

    def template_path(self, size: str) -> Path:
        raise NotImplementedError

    def convert_template(self, source: dict[str, Any], source_path: Path, size: str) -> dict[str, Any]:
        pieces_by_id = {piece["id"]: piece for piece in source["pieces"]}
        expected = PIECE_ORDER[self.style]
        pieces = {piece_id: self.convert_piece(pieces_by_id[piece_id]) for piece_id in expected}
        return {
            "size": self.body.sample_name,
            "garment_type": self.garment_type,
            "style": self.style,
            "source_template": relative_to_root(source_path),
            "source_version": source.get("template_version"),
            "pieces": pieces,
            "seams": SHORT_SEAMS if self.style == "short_sleeve" else LONG_SEAMS,
            "placement_rules": {piece_id: PLACEMENT_RULES[piece_id] for piece_id in expected},
            "fit": self.fit_summary(size),
        }

    def convert_piece(self, piece: dict[str, Any]) -> dict[str, Any]:
        scaled = self.scale_piece(piece)
        return {
            "role": piece.get("role"),
            "source_name": piece.get("source_name"),
            "is_main": piece.get("is_main"),
            "is_derived": piece.get("is_derived"),
            "symmetry": piece.get("symmetry"),
            "boundary_order": self.boundary_order(piece),
            "boundary": scaled["boundary"],
            "bbox": bbox(scaled["boundary"]),
            "points": scaled["points"],
            "curves": scaled["curves"],
            "semantic_edges": piece.get("semantic_edges", {}),
            "measurement_refs": piece.get("measurement_refs", {}),
            "reference_lines": piece.get("reference_lines", {}),
            "mesh": None,
        }

    def scale_piece(self, piece: dict[str, Any]) -> dict[str, Any]:
        sx, sy = self.piece_scale(piece["id"])
        source_boundary = deepcopy(piece.get("boundary", []))
        box = bbox(source_boundary)
        origin = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        boundary = scale_points(source_boundary, sx, sy, origin)
        points = {k: scale_points([deepcopy(v)], sx, sy, origin)[0] for k, v in piece.get("control_points", {}).items()}
        curves: dict[str, list[dict[str, Any]]] = {}
        for edge in piece.get("edges", []):
            role = edge.get("semantic_role")
            if not role:
                continue
            item = {key: deepcopy(edge.get(key)) for key in ("id", "start", "end", "length", "geometry_type")}
            item["polyline"] = scale_points(deepcopy(edge.get("polyline", [])), sx, sy, origin)
            curves.setdefault(role, []).append(item)
        return {"boundary": boundary, "points": points, "curves": curves}

    def piece_scale(self, piece_id: str) -> tuple[float, float]:
        body = self.effective_body_input
        ref = self.size_refs[self.select_size()]
        width = body["chest"] / float(ref["chest"])
        height = body["height"] / float(ref["height"])
        arm = body["arm_length"] / float(ref["arm_length"])
        if "sleeve" in piece_id or piece_id in {"cuff", "placket_tip", "placket_strip"}:
            return width, arm
        if "collar" in piece_id:
            return width, 1.0
        return width, height

    def boundary_order(self, piece: dict[str, Any]) -> list[str]:
        roles = []
        for edge in piece.get("edges", []):
            role = edge.get("semantic_role")
            if role and role not in roles:
                roles.append(role)
        preferred = BOUNDARY_ORDERS.get(piece["id"], roles)
        return [role for role in preferred if role in set(roles)] + [role for role in roles if role not in preferred]

    def fit_summary(self, size: str) -> dict[str, Any]:
        return {
            "mode": "capacity_template_scaled",
            "nearest_size": size,
            "body_input": self.effective_body_input,
            "effective_body_input": self.effective_body_input,
            "requested_body_input": self.requested_body_input,
            "input_resolution": self.input_resolution,
            "selection": self.selection,
        }
