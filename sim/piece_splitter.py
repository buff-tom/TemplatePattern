from __future__ import annotations

from copy import deepcopy
from typing import Any

from .curve_ops import (
    build_piece,
    clean_polyline,
    distance,
    nearest_index,
    point_at_length,
    polyline_between,
    polyline_length,
    project_to_x,
    same_point,
    semantic_polyline,
    split_at_length,
    split_at_x,
)


RIGHT_MAP = {
    "front_body_left": "front_body_right",
    "front_body_right": "front_body_left",
    "short_sleeve_front": "short_sleeve_front_right",
    "short_sleeve_back": "short_sleeve_back_right",
    "long_sleeve_front": "long_sleeve_front_right",
    "long_sleeve_back": "long_sleeve_back_right",
    "cuff_front": "cuff_front_right",
    "cuff_back": "cuff_back_right",
    "collar_stand_left": "collar_stand_right",
    "collar_stand_right": "collar_stand_left",
    "collar_fall_left": "collar_fall_right",
    "collar_fall_right": "collar_fall_left",
}

PANEL_FILTERS = {
    "long_sleeve": {
        "front_body_right", "front_body_left", "back_body", "back_yoke",
        "long_sleeve_front", "long_sleeve_back", "long_sleeve_front_right", "long_sleeve_back_right",
        "collar_stand_left", "collar_stand_back", "collar_stand_right",
        "collar_fall_left", "collar_fall_back", "collar_fall_right",
        "cuff_front", "cuff_back", "cuff_front_right", "cuff_back_right",
    },
    "short_sleeve": {
        "front_body_right", "front_body_left", "back_body", "back_yoke",
        "short_sleeve_front", "short_sleeve_back", "short_sleeve_front_right", "short_sleeve_back_right",
        "collar_stand_left", "collar_stand_back", "collar_stand_right",
        "collar_fall_left", "collar_fall_back", "collar_fall_right",
    },
}


class PieceSplitter:
    def expand(self, pattern: dict[str, Any]) -> dict[str, Any]:
        pattern = deepcopy(pattern)
        style = str(pattern.get("style"))
        pieces = pattern["pieces"]
        if style == "short_sleeve" and "short_sleeve" in pieces:
            pieces.update(self._split_sleeve(pieces["short_sleeve"], pieces, "short_sleeve"))
        if style == "long_sleeve" and "long_sleeve" in pieces:
            sleeve_pieces, meta = self._split_long_sleeve(pieces["long_sleeve"], pieces)
            pieces.update(sleeve_pieces)
            if "cuff" in pieces:
                pieces.update(self._split_cuff(pieces["cuff"], meta))
        if "collar_stand" in pieces:
            pieces.update(self._split_collar_stand(pieces["collar_stand"], pieces))
        if "collar_fall" in pieces:
            pieces.update(self._split_collar_fall(pieces["collar_fall"], pieces))
        pattern["pieces"] = {k: v for k, v in pieces.items() if k in PANEL_FILTERS[style]}
        pattern["seams"] = self._expanded_seams(pattern.get("seams", []), style)
        for seam in pattern['seams']:
            roles = [seam[s]['edge'] for s in ('a', 'b')]
            if roles == ['yoke_join', 'yoke_join']:
                axis = 0
            elif 'collar_stand_bottom' in roles and any('neckline' in role for role in roles):
                axis = 0
            elif roles == ['collar_stand_top', 'collar_fall_bottom']:
                axis = 0
            elif all(any(token in role for token in ('shoulder', 'side_seam', 'underarm', 'split_cut', 'sleeve_cap', 'armhole')) for role in roles):
                axis = 1
            else:
                continue
            lines = [semantic_polyline(pattern['pieces'][seam[s]['piece']], seam[s]['edge']) for s in ('a', 'b')]
            delta = [line[-1][axis]-line[0][axis] for line in lines]
            # Templates do not share traversal directions across styles/sizes.
            # Match upper endpoints to upper endpoints (or left to left for
            # the back join), independent of how the source curve was stored.
            if abs(delta[0]) > 1e-6 and abs(delta[1]) > 1e-6:
                seam['direction'] = 'same' if delta[0]*delta[1] > 0 else 'opposite'
        return pattern

    def _split_sleeve(self, piece: dict[str, Any], pieces: dict[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
        cap = semantic_polyline(piece, "sleeve_cap")
        front_under = semantic_polyline(piece, "underarm_front")
        back_under = semantic_polyline(piece, "underarm_back")
        hem_name = "cuff" if prefix == "short_sleeve" else "sleeve_cuff_join"
        hem = semantic_polyline(piece, hem_name)
        top = piece.get("points", {}).get("sleeve_cap_top") or cap[len(cap) // 2]
        split_index = nearest_index(cap, top)
        split_point = cap[split_index]
        front_cap = cap[: split_index + 1]
        back_cap = cap[split_index:]
        yoke_cap, body_cap = self._split_back_cap(back_cap, pieces)
        hem_point = project_to_x(split_point, hem)
        front = build_piece(
            piece,
            f"{prefix}_front",
            ["underarm_front", "sleeve_cap_front", "split_cut", "hem_front"],
            {"underarm_front": front_under, "sleeve_cap_front": front_cap, "split_cut": [split_point, hem_point], "hem_front": [hem_point, front_under[0]]},
            f"{prefix} front split at sleeve_cap_top",
        )
        back = build_piece(
            piece,
            f"{prefix}_back",
            ["split_cut", "sleeve_cap_yoke", "sleeve_cap_back_body", "underarm_back", "hem_back"],
            {"split_cut": [hem_point, split_point], "sleeve_cap_yoke": yoke_cap, "sleeve_cap_back_body": body_cap, "underarm_back": back_under, "hem_back": [back_under[-1], hem_point]},
            f"{prefix} back split at sleeve_cap_top",
        )
        return {front["id"]: front, back["id"]: back, f"{front['id']}_right": self._mirror(front, f"{front['id']}_right"), f"{back['id']}_right": self._mirror(back, f"{back['id']}_right")}

    def _split_long_sleeve(self, piece: dict[str, Any], pieces: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        split = self._split_sleeve(piece, pieces, "long_sleeve")
        cuff = semantic_polyline(piece, "sleeve_cuff_join")
        front_join = semantic_polyline(split["long_sleeve_front"], "hem_front")
        ratio = polyline_length(front_join) / max(polyline_length(cuff), 1.0e-9)
        for item in split.values():
            item["curves"]["sleeve_cuff_join"] = item["curves"].pop("hem_front", item["curves"].pop("hem_back", []))
            item["semantic_edges"]["sleeve_cuff_join"] = item["semantic_edges"].pop("hem_front", item["semantic_edges"].pop("hem_back", []))
            item["boundary_order"] = ["underarm_front", "sleeve_cap_front", "split_cut", "sleeve_cuff_join"] if "front" in item["id"] else ["split_cut", "sleeve_cap_yoke", "sleeve_cap_back_body", "underarm_back", "sleeve_cuff_join"]
        return split, {"cuff_top_front_ratio": ratio}

    def _split_back_cap(self, back_cap: list[list[float]], pieces: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
        yoke = polyline_length(semantic_polyline(pieces.get("back_yoke", {}), "armhole_left_part"))
        body = polyline_length(semantic_polyline(pieces.get("back_body", {}), "armhole_left_part"))
        target = polyline_length(back_cap) * (0.5 if yoke + body <= 1.0e-9 else yoke / (yoke + body))
        return split_at_length(back_cap, target)

    def _split_cuff(self, piece: dict[str, Any], meta: dict[str, float]) -> dict[str, dict[str, Any]]:
        top = semantic_polyline(piece, "cuff_top_join")
        bottom = semantic_polyline(piece, "cuff_bottom_edge")
        short = piece.get("curves", {}).get("cuff_short_end", [])
        right = clean_polyline(short[0].get("polyline", [])) if short else []
        left = clean_polyline(short[-1].get("polyline", [])) if short else []
        top_front, top_back = split_at_length(top, polyline_length(top) * float(meta.get("cuff_top_front_ratio", 0.5)))
        bottom_back, bottom_front = split_at_length(bottom, polyline_length(bottom) * (1.0 - float(meta.get("cuff_top_front_ratio", 0.5))))
        cut = [top_front[-1], bottom_front[0]]
        front = build_piece(piece, "cuff_front", ["cuff_top_join", "split_cut", "cuff_bottom_edge", "cuff_short_end_left"], {"cuff_top_join": top_front, "split_cut": cut, "cuff_bottom_edge": bottom_front, "cuff_short_end_left": left}, "front cuff split by sleeve cuff ratio")
        back = build_piece(piece, "cuff_back", ["cuff_top_join", "cuff_short_end_right", "cuff_bottom_edge", "split_cut"], {"cuff_top_join": top_back, "cuff_short_end_right": right, "cuff_bottom_edge": bottom_back, "split_cut": list(reversed(cut))}, "back cuff split by sleeve cuff ratio")
        return {"cuff_front": front, "cuff_back": back, "cuff_front_right": self._mirror(front, "cuff_front_right"), "cuff_back_right": self._mirror(back, "cuff_back_right")}

    def _split_collar_stand(self, piece: dict[str, Any], pieces: dict[str, Any]) -> dict[str, dict[str, Any]]:
        bottom = semantic_polyline(piece, "collar_stand_bottom")
        top = semantic_polyline(piece, "collar_stand_top")
        top = self._align_strip(bottom, top)
        left, back, right = self._three_way(bottom, pieces)
        top_left, top_back, top_right = self._three_way(top, pieces)
        return {
            "collar_stand_left": self._collar_piece(piece, "collar_stand_left", left, top_left, "collar_stand"),
            "collar_stand_back": self._collar_piece(piece, "collar_stand_back", back, top_back, "collar_stand"),
            "collar_stand_right": self._collar_piece(piece, "collar_stand_right", right, top_right, "collar_stand"),
        }

    def _split_collar_fall(self, piece: dict[str, Any], pieces: dict[str, Any]) -> dict[str, dict[str, Any]]:
        bottom = semantic_polyline(piece, "collar_fall_bottom")
        top = semantic_polyline(piece, "collar_fall_top")
        top = self._align_strip(bottom, top)
        left, back, right = self._three_way(bottom, pieces)
        top_left, top_back, top_right = self._three_way(top, pieces)
        return {
            "collar_fall_left": self._collar_piece(piece, "collar_fall_left", left, top_left, "collar_fall"),
            "collar_fall_back": self._collar_piece(piece, "collar_fall_back", back, top_back, "collar_fall"),
            "collar_fall_right": self._collar_piece(piece, "collar_fall_right", right, top_right, "collar_fall"),
        }

    def _three_way(self, polyline: list[list[float]], pieces: dict[str, Any]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        lengths = [polyline_length(semantic_polyline(pieces.get("front_body_left", {}), "neckline")), polyline_length(semantic_polyline(pieces.get("back_yoke", {}), "back_neckline")), polyline_length(semantic_polyline(pieces.get("front_body_right", {}), "neckline"))]
        total = sum(lengths) or 3.0
        left, rest = split_at_length(polyline, polyline_length(polyline) * lengths[0] / total)
        back, right = split_at_length(rest, polyline_length(rest) * lengths[1] / max(total - lengths[0], 1.0e-9))
        return left, back, right

    @staticmethod
    def _align_strip(bottom, top):
        # Source boundary loops traverse top and bottom in opposite directions.
        # Split both in the same spatial direction, otherwise cut edges cross.
        from math import dist
        same = dist(bottom[0], top[0]) + dist(bottom[-1], top[-1])
        reverse = dist(bottom[0], top[-1]) + dist(bottom[-1], top[0])
        return list(reversed(top)) if reverse < same else top

    def _collar_piece(self, source: dict[str, Any], name: str, bottom: list[list[float]], top: list[list[float]], prefix: str) -> dict[str, Any]:
        return build_piece(source, name, [f"{prefix}_bottom", "cut_right", f"{prefix}_top", "cut_left"], {f"{prefix}_bottom": bottom, "cut_right": [bottom[-1], top[-1]], f"{prefix}_top": list(reversed(top)), "cut_left": [top[0], bottom[0]]}, f"{name} split from {prefix}")

    def _mirror(self, piece: dict[str, Any], name: str) -> dict[str, Any]:
        clone = deepcopy(piece)
        clone["id"] = name
        clone["name"] = name
        clone["boundary"] = [[-p[0], p[1]] for p in clone.get("boundary", [])]
        clone["points"] = {k: [-v[0], v[1]] for k, v in clone.get("points", {}).items()}
        for entries in clone.get("curves", {}).values():
            for entry in entries:
                entry["polyline"] = [[-p[0], p[1]] for p in entry.get("polyline", [])]
        return clone

    def _expanded_seams(self, seams: list[dict[str, Any]], style: str) -> list[dict[str, Any]]:
        out = [s for s in seams if s.get("target") == "sew" and s["a"]["piece"] in PANEL_FILTERS[style] and s["b"]["piece"] in PANEL_FILTERS[style]]
        out += self._collar_seams()
        out += self._sleeve_seams(style)
        return [s for s in out if s["a"]["piece"] in PANEL_FILTERS[style] and s["b"]["piece"] in PANEL_FILTERS[style]]

    def _collar_seams(self) -> list[dict[str, Any]]:
        return [
            self._seam("collar_stand_left", "collar_stand_bottom", "front_body_left", "neckline"),
            self._seam("collar_stand_back", "collar_stand_bottom", "back_yoke", "back_neckline"),
            self._seam("collar_stand_right", "collar_stand_bottom", "front_body_right", "neckline"),
            self._seam("collar_stand_left", "collar_stand_top", "collar_fall_left", "collar_fall_bottom"),
            self._seam("collar_stand_back", "collar_stand_top", "collar_fall_back", "collar_fall_bottom"),
            self._seam("collar_stand_right", "collar_stand_top", "collar_fall_right", "collar_fall_bottom"),
            self._seam("collar_stand_left", "cut_right", "collar_stand_back", "cut_left"),
            self._seam("collar_stand_back", "cut_right", "collar_stand_right", "cut_left"),
            self._seam("collar_fall_left", "cut_right", "collar_fall_back", "cut_left"),
            self._seam("collar_fall_back", "cut_right", "collar_fall_right", "cut_left"),
        ]

    def _sleeve_seams(self, style: str) -> list[dict[str, Any]]:
        prefix = "short_sleeve" if style == "short_sleeve" else "long_sleeve"
        seams = [self._seam(f"{prefix}_front", "sleeve_cap_front", "front_body_left", "armhole"), self._seam(f"{prefix}_back", "sleeve_cap_yoke", "back_yoke", "armhole_left_part"), self._seam(f"{prefix}_back", "sleeve_cap_back_body", "back_body", "armhole_left_part"), self._seam(f"{prefix}_front", "split_cut", f"{prefix}_back", "split_cut"), self._seam(f"{prefix}_front", "underarm_front", f"{prefix}_back", "underarm_back")]
        # Front cap and front armhole both run underarm -> shoulder;
        # back cap/yoke both run shoulder -> lower armhole. Reversing either
        # pair twists the sleeve by sewing shoulder to underarm.
        seams[0]['direction'] = 'same'
        seams[1]['direction'] = 'same'
        seams += [self._mirror_seam(s) for s in seams]
        if style == "long_sleeve":
            seams += [self._seam("long_sleeve_front", "sleeve_cuff_join", "cuff_front", "cuff_top_join"), self._seam("long_sleeve_back", "sleeve_cuff_join", "cuff_back", "cuff_top_join"), self._seam("cuff_front", "split_cut", "cuff_back", "split_cut")]
            seams += [self._mirror_seam(s) for s in seams[-3:]]
        return seams

    def _mirror_seam(self, seam: dict[str, Any]) -> dict[str, Any]:
        seam = deepcopy(seam)
        for side in ("a", "b"):
            seam[side]["piece"] = RIGHT_MAP.get(seam[side]["piece"], seam[side]["piece"])
            seam[side]["edge"] = seam[side]["edge"].replace("_left", "_SIDE_SWAP").replace("_right", "_left").replace("_SIDE_SWAP", "_right")
        return seam

    def _seam(self, a_piece: str, a_edge: str, b_piece: str, b_edge: str) -> dict[str, Any]:
        return {"a": {"piece": a_piece, "edge": a_edge}, "b": {"piece": b_piece, "edge": b_edge}, "direction": "opposite", "target": "sew"}
