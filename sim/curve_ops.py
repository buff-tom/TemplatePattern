from __future__ import annotations

import math
from typing import Any


Point = list[float]


def same_point(a: Point, b: Point, tol: float = 1.0e-5) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clean_polyline(polyline: list[Point]) -> list[Point]:
    out: list[Point] = []
    for point in polyline:
        p = [float(point[0]), float(point[1])]
        if not out or not same_point(out[-1], p):
            out.append(p)
    if len(out) > 1 and same_point(out[0], out[-1]):
        out.pop()
    return out


def polyline_length(polyline: list[Point]) -> float:
    return sum(distance(a, b) for a, b in zip(polyline, polyline[1:]))


def semantic_polyline(piece: dict[str, Any], name: str) -> list[Point]:
    points: list[Point] = []
    for curve in piece.get("curves", {}).get(name, []):
        polyline = clean_polyline(curve.get("polyline", []))
        if not polyline:
            continue
        if points and same_point(points[-1], polyline[0]):
            points.extend(polyline[1:])
        else:
            points.extend(polyline)
    return clean_polyline(points)


def point_at_length(polyline: list[Point], target: float) -> Point:
    if not polyline:
        return [0.0, 0.0]
    if target <= 0:
        return list(polyline[0])
    walked = 0.0
    for a, b in zip(polyline, polyline[1:]):
        seg = distance(a, b)
        if walked + seg >= target and seg > 1.0e-9:
            ratio = (target - walked) / seg
            return [a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio]
        walked += seg
    return list(polyline[-1])


def split_at_length(polyline: list[Point], target: float) -> tuple[list[Point], list[Point]]:
    polyline = clean_polyline(polyline)
    if len(polyline) < 2:
        return polyline, polyline
    point = point_at_length(polyline, target)
    walked = 0.0
    left = [polyline[0]]
    for index, (a, b) in enumerate(zip(polyline, polyline[1:])):
        seg = distance(a, b)
        if walked + seg >= target:
            if not same_point(left[-1], point):
                left.append(point)
            right = [point]
            if not same_point(point, b):
                right.append(b)
            right.extend(polyline[index + 2 :])
            return clean_polyline(left), clean_polyline(right)
        if not same_point(left[-1], b):
            left.append(b)
        walked += seg
    return polyline, [polyline[-1]]


def split_at_x(polyline: list[Point], target_x: float) -> tuple[list[Point], list[Point]]:
    polyline = clean_polyline(polyline)
    if len(polyline) < 2:
        return polyline, polyline
    best = min(range(len(polyline)), key=lambda i: abs(polyline[i][0] - target_x))
    if 0 < best < len(polyline) - 1:
        return polyline[: best + 1], polyline[best:]
    for index, (a, b) in enumerate(zip(polyline, polyline[1:])):
        if (a[0] - target_x) * (b[0] - target_x) <= 0 and abs(a[0] - b[0]) > 1.0e-9:
            ratio = (target_x - a[0]) / (b[0] - a[0])
            p = [target_x, a[1] + (b[1] - a[1]) * ratio]
            return clean_polyline(polyline[: index + 1] + [p]), clean_polyline([p] + polyline[index + 1 :])
    return polyline[:1], polyline


def nearest_index(polyline: list[Point], point: Point) -> int:
    return min(range(len(polyline)), key=lambda i: distance(polyline[i], point))


def polyline_between(polyline: list[Point], start: Point, end: Point) -> list[Point]:
    polyline = clean_polyline(polyline)
    if len(polyline) < 2:
        return polyline
    i = nearest_index(polyline, start)
    j = nearest_index(polyline, end)
    if i <= j:
        return clean_polyline([start] + polyline[i + 1 : j] + [end])
    return clean_polyline([start] + list(reversed(polyline[j + 1 : i])) + [end])


def project_to_x(point: Point, polyline: list[Point]) -> Point:
    if not polyline:
        return list(point)
    return min(polyline, key=lambda p: abs(p[0] - point[0]))


def build_piece(source: dict[str, Any], name: str, order: list[str], curves: dict[str, list[Point]], note: str) -> dict[str, Any]:
    boundary: list[Point] = []
    curve_payload: dict[str, list[dict[str, Any]]] = {}
    for role in order:
        polyline = clean_polyline(curves.get(role, []))
        if len(polyline) < 2:
            continue
        if boundary and same_point(boundary[-1], polyline[0]):
            boundary.extend(polyline[1:])
        else:
            boundary.extend(polyline)
        curve_payload[role] = [{"id": f"{name}.{role}.01", "polyline": polyline}]
    boundary = clean_polyline(boundary)
    xs = [p[0] for p in boundary] or [0.0]
    ys = [p[1] for p in boundary] or [0.0]
    piece = {k: v for k, v in source.items() if k not in {"id", "curves", "boundary", "bbox", "semantic_edges"}}
    piece.update(
        {
            "id": name,
            "name": name,
            "curves": curve_payload,
            "boundary_order": order,
            "boundary": boundary,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "semantic_edges": {role: [f"{name}.{role}.01"] for role in curve_payload},
            "generation_note": note,
            "preserve_boundary_order": True,
        }
    )
    return piece
