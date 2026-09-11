from __future__ import annotations

import math
from typing import Iterable


Point = list[float]


def bbox(points: Iterable[Point]) -> list[float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def center_box(box: list[float]) -> Point:
    return [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(polyline: list[Point]) -> float:
    return sum(distance(a, b) for a, b in zip(polyline, polyline[1:]))


def normalize_points(points: list[Point]) -> list[Point]:
    box = bbox(points)
    cx, cy = center_box(box)
    return [[p[0] - cx, p[1] - cy] for p in points]


def scale_points(points: list[Point], sx: float, sy: float, origin: Point | None = None) -> list[Point]:
    ox, oy = origin or center_box(bbox(points))
    return [[ox + (p[0] - ox) * sx, oy + (p[1] - oy) * sy] for p in points]


def translate_points(points: list[Point], dx: float, dy: float) -> list[Point]:
    return [[p[0] + dx, p[1] + dy] for p in points]


def sequential_edges(count: int) -> list[dict[str, list[int]]]:
    return [{"endpoints": [i, (i + 1) % count]} for i in range(count)]


def nearest_index(points: list[Point], target: Point) -> int:
    return min(range(len(points)), key=lambda i: distance(points[i], target))


def edge_ids_between(start: int, end: int, count: int) -> list[int]:
    if count <= 0:
        return []
    ids = []
    i = start
    while i != end:
        ids.append(i)
        i = (i + 1) % count
        if len(ids) > count:
            break
    return ids or [start % count]

