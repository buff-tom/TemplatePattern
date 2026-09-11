from __future__ import annotations

from math import cos, hypot, radians
from typing import Any


Point = list[float]


def sample_boundary(piece: dict[str, Any], target_edge_length: float = 10.0) -> list[Point]:
    boundary = _dedupe([(float(point[0]), float(point[1])) for point in piece.get("boundary", [])])
    if len(boundary) > 1 and _distance(boundary[0], boundary[-1]) < 1.0e-7:
        boundary.pop()
    if len(boundary) < 2:
        return [[point[0], point[1]] for point in boundary]
    protected = _protected_indices(piece, boundary)
    sampled = _resample_closed(boundary, protected, target_edge_length)
    return [[round(point[0], 6), round(point[1], 6)] for point in sampled]


def _protected_indices(piece: dict[str, Any], boundary: list[tuple[float, float]]) -> list[int]:
    endpoints = [
        (float(point[0]), float(point[1]))
        for curves in piece.get("curves", {}).values()
        for curve in curves
        for point in ((curve.get("polyline") or [])[:1] + (curve.get("polyline") or [])[-1:])
    ]
    protected = {0}
    for index, point in enumerate(boundary):
        if any(_distance(point, endpoint) < 1.0e-7 for endpoint in endpoints):
            protected.add(index)
        if _is_sharp(boundary[index - 1], point, boundary[(index + 1) % len(boundary)]):
            protected.add(index)
    return sorted(protected)


def _is_sharp(previous: tuple[float, float], current: tuple[float, float], following: tuple[float, float]) -> bool:
    incoming = (current[0] - previous[0], current[1] - previous[1])
    outgoing = (following[0] - current[0], following[1] - current[1])
    incoming_length = hypot(*incoming)
    outgoing_length = hypot(*outgoing)
    if incoming_length < 1.0e-7 or outgoing_length < 1.0e-7:
        return False
    direction_cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_length * outgoing_length)
    return direction_cosine < cos(radians(20.0))


def _resample_closed(
    points: list[tuple[float, float]],
    protected_indices: list[int],
    target_edge_length: float,
) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for offset, start_index in enumerate(protected_indices):
        end_index = protected_indices[(offset + 1) % len(protected_indices)]
        section = points[start_index : end_index + 1] if end_index > start_index else points[start_index:] + points[: end_index + 1]
        sampled.extend(_resample_polyline(section, target_edge_length)[:-1])
    return _dedupe(sampled)


def _resample_polyline(points: list[tuple[float, float]], target_edge_length: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(start, end))
    total_length = cumulative[-1]
    if total_length < 1.0e-7:
        return [points[0]]
    sample_count = max(1, round(total_length / target_edge_length))
    sample_interval = total_length / sample_count
    sampled = [points[0]]
    sample_distance = sample_interval
    segment_index = 0
    while sample_distance < total_length:
        while cumulative[segment_index + 1] < sample_distance:
            segment_index += 1
        start = points[segment_index]
        end = points[segment_index + 1]
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
        ratio = (sample_distance - cumulative[segment_index]) / segment_length
        sampled.append((start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio))
        sample_distance += sample_interval
    if _distance(sampled[-1], points[-1]) > 1.0e-7:
        sampled.append(points[-1])
    return sampled


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _dedupe(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or _distance(deduped[-1], point) >= 1.0e-7:
            deduped.append(point)
    return deduped
