from __future__ import annotations

from collections import Counter, defaultdict
from math import hypot
from typing import Any

from .curve_ops import clean_polyline, semantic_polyline


Point = list[float]
EdgeRef = tuple[str, str]

START_POINT_ALIASES = {
    ("collar_stand", "collar_stand_top"): ("stand_top_peak_left",),
    ("collar_fall", "collar_fall_bottom"): ("collar_tip_left",),
    ("front_body_left", "neckline"): ("cf_top_left", "cf_top"),
    ("back_yoke", "back_neckline"): ("back_neck_point_left", "neck_point_left"),
    ("front_body_right", "neckline"): ("front_neck_point_right", "neck_point"),
}


class StitchBuilder:
    def build(self, pattern: dict[str, Any], edge_lookup: dict[EdgeRef, list[int]]) -> tuple[list[Any], list[dict[str, Any]]]:
        seams = [seam for seam in pattern.get("seams", []) if seam.get("target") == "sew"]
        seam_refs = Counter((seam["a"]["piece"], seam["a"]["edge"]) for seam in seams)
        seam_refs.update((seam["b"]["piece"], seam["b"]["edge"]) for seam in seams)
        edge_lengths = self._edge_lengths(pattern)
        allocated = self._allocate_reused_edges(seams, edge_lookup, edge_lengths)
        stitches: list[Any] = []
        skipped: list[dict[str, Any]] = []
        used_edges: set[tuple[str, int]] = set()
        for seam_index, seam in enumerate(seams):
            key_a = (seam["a"]["piece"], seam["a"]["edge"])
            key_b = (seam["b"]["piece"], seam["b"]["edge"])
            group_a = allocated.get((seam_index, "a"), [])
            group_b = allocated.get((seam_index, "b"), [])
            if not group_a or not group_b:
                skipped.append({"reason": "missing_edge", "a": key_a, "b": key_b})
                continue
            if seam.get("direction", "same") == "opposite":
                group_b = list(reversed(group_b))
            for pair_index, (edge_a, edge_b) in enumerate(self._pair_vertices(group_a, group_b)):
                if self._skip_endpoint_pairing(key_a, key_b, pair_index, len(group_a), len(group_b)):
                    continue
                stitch_keys = {(key_a[0], edge_a), (key_b[0], edge_b)}
                if stitch_keys & used_edges:
                    skipped.append({"reason": "panel_edge_already_stitched", "a": key_a, "b": key_b, "pair": [edge_a, edge_b]})
                    continue
                stitch: list[Any] = [{"panel": key_a[0], "edge": edge_a}, {"panel": key_b[0], "edge": edge_b}]
                if seam.get("direction", "same") == "same":
                    stitch.append("right_wrong")
                stitches.append(stitch)
                used_edges.update(stitch_keys)
        return stitches, skipped

    def lookup_for_piece(self, piece_name: str, piece: dict[str, Any], vertices: list[list[float]]) -> dict[EdgeRef, list[int]]:
        edge_polylines = self._edge_polylines(piece)
        if not edge_polylines or len(vertices) < 2:
            return {}
        assignments: dict[str, list[int]] = defaultdict(list)
        for local_index, point in enumerate(vertices):
            edge_name = min(edge_polylines, key=lambda name: self._distance_to_polylines(point, edge_polylines[name]))
            assignments[edge_name].append(local_index)
        lookup: dict[EdgeRef, list[int]] = {}
        for edge_name, polylines in edge_polylines.items():
            indices = self._include_polyline_endpoints(assignments.get(edge_name, []), vertices, polylines)
            if len(indices) < 2:
                continue
            ordered = sorted(set(indices), key=lambda index: self._progress_along_polylines(vertices[index], polylines))
            lookup[(piece_name, edge_name)] = self._orient_edge_vertices(piece_name, edge_name, ordered, vertices, piece.get("points", {}))
        return lookup

    def _edge_lengths(self, pattern: dict[str, Any]) -> dict[EdgeRef, float]:
        lengths: dict[EdgeRef, float] = {}
        for piece_name, piece in pattern.get("pieces", {}).items():
            for edge_name in piece.get("curves", {}):
                lengths[(piece_name, edge_name)] = self._polyline_length(semantic_polyline(piece, edge_name))
        return lengths

    def _allocate_reused_edges(
        self,
        seams: list[dict[str, Any]],
        edge_lookup: dict[EdgeRef, list[int]],
        edge_lengths: dict[EdgeRef, float],
    ) -> dict[tuple[int, str], list[int]]:
        usages: dict[EdgeRef, list[tuple[int, str]]] = defaultdict(list)
        for seam_index, seam in enumerate(seams):
            usages[(seam["a"]["piece"], seam["a"]["edge"])].append((seam_index, "a"))
            usages[(seam["b"]["piece"], seam["b"]["edge"])].append((seam_index, "b"))
        allocated: dict[tuple[int, str], list[int]] = {}
        for edge_ref, edge_usages in usages.items():
            vertices = edge_lookup.get(edge_ref, [])
            if not vertices:
                continue
            if len(edge_usages) == 1:
                allocated[edge_usages[0]] = vertices
                continue
            weights = []
            for seam_index, side in edge_usages:
                other_side = "b" if side == "a" else "a"
                other = seams[seam_index][other_side]
                weights.append(edge_lengths.get((other["piece"], other["edge"]), 1.0))
            for usage, vertex_slice in zip(edge_usages, self._partition_vertices(vertices, weights)):
                allocated[usage] = vertex_slice
        return allocated

    def _partition_vertices(self, vertices: list[int], weights: list[float]) -> list[list[int]]:
        total_weight = sum(weights)
        if total_weight <= 0.0:
            weights = [1.0] * len(weights)
            total_weight = float(len(weights))
        partitions: list[list[int]] = []
        start = 0
        accumulated = 0.0
        segment_count = len(vertices) - 1
        for index, weight in enumerate(weights):
            accumulated += weight
            end = segment_count if index == len(weights) - 1 else round(segment_count * accumulated / total_weight)
            end = max(start + 1, min(segment_count, end))
            partitions.append(vertices[start : end + 1])
            start = end
        return partitions

    def _pair_vertices(self, a_vertices: list[int], b_vertices: list[int]) -> list[tuple[int, int]]:
        pair_count = max(len(a_vertices), len(b_vertices))
        if pair_count < 2:
            return []
        pairs: list[tuple[int, int]] = []
        for index in range(pair_count):
            progress = index / (pair_count - 1)
            a = a_vertices[round(progress * (len(a_vertices) - 1))]
            b = b_vertices[round(progress * (len(b_vertices) - 1))]
            if not pairs or pairs[-1] != (a, b):
                pairs.append((a, b))
        return pairs

    def _skip_endpoint_pairing(
        self,
        key_a: EdgeRef,
        key_b: EdgeRef,
        pair_index: int,
        len_a: int,
        len_b: int,
    ) -> bool:
        pair_count = max(len_a, len_b)
        if pair_count <= 2:
            return False
        both_split = key_a[1] == "split_cut" and key_b[1] == "split_cut"
        return both_split and pair_index in {0, pair_count - 1}

    def _edge_polylines(self, piece: dict[str, Any]) -> dict[str, list[list[Point]]]:
        resolved: dict[str, list[list[Point]]] = {}
        for edge_name, curves in piece.get("curves", {}).items():
            polylines = []
            for curve in curves:
                polyline = clean_polyline(curve.get("polyline", []))
                if len(polyline) >= 2:
                    polylines.append(polyline)
            if polylines:
                resolved[edge_name] = polylines
        return resolved

    def _include_polyline_endpoints(
        self,
        indices: list[int],
        boundary: list[Point],
        polylines: list[list[Point]],
    ) -> list[int]:
        resolved = set(indices)
        for polyline in polylines:
            for endpoint in (polyline[0], polyline[-1]):
                local_index = min(range(len(boundary)), key=lambda index: self._distance(boundary[index], endpoint))
                if self._distance(boundary[local_index], endpoint) <= 1.0e-6:
                    resolved.add(local_index)
        return list(resolved)

    def _orient_edge_vertices(
        self,
        piece_name: str,
        edge_name: str,
        vertices: list[int],
        boundary: list[Point],
        points: dict[str, Point],
    ) -> list[int]:
        named_starts = START_POINT_ALIASES.get((piece_name, edge_name))
        if (piece_name, edge_name) == ("collar_stand", "collar_stand_bottom"):
            named_starts = ("right_end",)
        if not named_starts:
            return vertices
        start = next((points[name] for name in named_starts if name in points), None)
        if not start:
            return vertices
        first = boundary[vertices[0]]
        last = boundary[vertices[-1]]
        return list(reversed(vertices)) if self._distance(last, start) < self._distance(first, start) else vertices

    def _distance_to_polylines(self, point: Point, polylines: list[list[Point]]) -> float:
        return min(self._distance_to_polyline(point, polyline) for polyline in polylines)

    def _distance_to_polyline(self, point: Point, polyline: list[Point]) -> float:
        best = float("inf")
        for start, end in zip(polyline, polyline[1:]):
            best = min(best, self._distance_to_segment(point, start, end))
        return best

    def _distance_to_segment(self, point: Point, start: Point, end: Point) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        span = dx * dx + dy * dy
        if span <= 1.0e-12:
            return self._distance(point, start)
        ratio = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / span
        ratio = max(0.0, min(1.0, ratio))
        proj = [start[0] + dx * ratio, start[1] + dy * ratio]
        return self._distance(point, proj)

    def _progress_along_polylines(self, point: Point, polylines: list[list[Point]]) -> float:
        best_progress = 0.0
        best_distance = float("inf")
        offset = 0.0
        total = sum(self._polyline_length(polyline) for polyline in polylines) or 1.0
        for polyline in polylines:
            walked = 0.0
            for start, end in zip(polyline, polyline[1:]):
                dist, ratio = self._segment_projection(point, start, end)
                if dist < best_distance:
                    best_distance = dist
                    best_progress = (offset + walked + self._distance(start, end) * ratio) / total
                walked += self._distance(start, end)
            offset += walked
        return best_progress

    def _segment_projection(self, point: Point, start: Point, end: Point) -> tuple[float, float]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        span = dx * dx + dy * dy
        if span <= 1.0e-12:
            return self._distance(point, start), 0.0
        ratio = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / span
        ratio = max(0.0, min(1.0, ratio))
        proj = [start[0] + dx * ratio, start[1] + dy * ratio]
        return self._distance(point, proj), ratio

    def _polyline_length(self, polyline: list[Point]) -> float:
        return sum(self._distance(a, b) for a, b in zip(polyline, polyline[1:]))

    def _distance(self, a: Point, b: Point) -> float:
        return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
