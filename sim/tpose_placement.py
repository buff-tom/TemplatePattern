from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from TemplatePattern_final.shared.io import read_json, write_json


LONG_TORSO_PRESET = {
    "front_body_right": ([-13.5, 110.0, 30.0], [0.0, 0.0, 0.0]),
    "front_body_left": ([12.5, 110.0, 30.0], [0.0, 0.0, 0.0]),
    "back_body": ([0.0, 110.0, -30.0], [0.0, 180.0, 0.0]),
    "back_yoke": ([0.0, 148.5, -30.5], [0.0, 180.0, 0.0]),
    "collar_stand_left": ([7.82, 148.56, 30.0], [0.0, 0.0, 180.0]),
    "collar_stand_back": ([0.0, 161.57, -30.5], [0.0, 0.0, 180.0]),
    "collar_stand_right": ([-7.82, 148.56, 30.0], [0.0, 0.0, 180.0]),
    "collar_fall_left": ([11.82, 157.56, 30.0], [0.0, 0.0, 180.0]),
    "collar_fall_back": ([0.0, 170.57, -30.5], [0.0, 0.0, 180.0]),
    "collar_fall_right": ([-11.82, 157.56, 30.0], [0.0, 0.0, 180.0]),
}

LONG_SLEEVE_FRONT_Z = 30.0
LONG_SLEEVE_BACK_Z = -30.0
LONG_CUFF_FRONT_Z = 30.0
LONG_CUFF_BACK_Z = -30.0
LONG_CUFF_PROXIMAL_OFFSET_CM = 3.0


class BodyPosePlacement:
    def apply_file(
        self,
        spec_path: str | Path,
        style: str,
        *,
        body_assets_dir: str | Path,
        body_name: str,
        pose: str = "a45",
        arm_angle_deg: float = 45.0,
        x_clearance: float = 2.0,
        wrist_clearance_cm: float = 0.45,
    ) -> dict[str, str]:
        spec = read_json(spec_path)
        landmarks = self._compute_landmarks(Path(body_assets_dir), body_name)
        relation = self.apply(spec, style, landmarks, pose, arm_angle_deg, x_clearance)
        if style == 'long_sleeve':
            from .wrist_placement import place_wrists
            body_vertices = self._load_obj_vertices(Path(body_assets_dir) / f'{body_name}.obj')
            try:
                place_wrists(spec, landmarks['wrists'], relation, wrist_clearance_cm, body_vertices)
            except ValueError as exc:
                relation['placement_error'] = str(exc)
                write_json(Path(spec_path).parent / 'body_pose_relation.json', relation)
                write_json(Path(body_assets_dir) / 'placement_landmarks.json', landmarks)
                raise
        relation["placement_debug"] = self._debug_summary(spec)
        write_json(spec_path, spec)

        spec_path = Path(spec_path)
        relation_path = spec_path.parent / "body_pose_relation.json"
        landmarks_path = Path(body_assets_dir) / "placement_landmarks.json"
        write_json(relation_path, relation)
        write_json(landmarks_path, landmarks)
        return {
            "spec": str(spec_path),
            "relation": str(relation_path),
            "body_landmarks": str(landmarks_path),
        }

    def apply(
        self,
        spec: dict[str, Any],
        style: str,
        landmarks: dict[str, Any],
        pose: str,
        arm_angle_deg: float,
        x_clearance: float,
    ) -> dict[str, Any]:
        relation = {
            "style": style,
            "pose": pose,
            "arm_angle_deg": float(arm_angle_deg),
            "x_clearance_cm": float(x_clearance),
            "piece_placement": {},
            "body_landmarks_cm": landmarks,
        }
        from .body_relative_placement import place_relative_to_body
        place_relative_to_body(spec, landmarks, style, pose, relation)
        return relation

    def _compute_landmarks(self, body_assets_dir: Path, body_name: str) -> dict[str, Any]:
        vertices = self._load_obj_vertices(body_assets_dir / f"{body_name}.obj")
        segmentation = read_json(body_assets_dir / "smpl_vert_segmentation.json")

        def center(name: str) -> list[float]:
            indices = segmentation[name]
            points = [vertices[index] for index in indices if index < len(vertices)]
            if not points:
                raise ValueError(f"body segmentation {name} has no valid vertices")
            return [round(sum(point[axis] for point in points) / len(points), 6) for axis in range(3)]

        def hand_root(side: str, forearm: list[float]) -> list[float]:
            points = [vertices[index] for index in segmentation[f"{side}Hand"] if index < len(vertices)]
            root = min(points, key=lambda point: self._dist3(point, forearm))
            return [round(value, 6) for value in root]

        def bounds(names):
            points = [vertices[i] for name in names for i in segmentation.get(name, [])]
            return [min(p[a] for p in points) for a in range(3)] + [max(p[a] for p in points) for a in range(3)]

        segments: dict[str, Any] = {}
        for side in ("left", "right"):
            shoulder = center(f"{side}Shoulder")
            upper_arm = center(f"{side}Arm")
            forearm = center(f"{side}ForeArm")
            root = hand_root(side, forearm)
            hand = center(f"{side}Hand")
            segments[side] = {
                "shoulder": shoulder,
                "upper_arm": upper_arm,
                "forearm": forearm,
                "hand_root": root,
                "hand": hand,
                "long_sleeve_center": self._lerp(shoulder, root, 0.48),
                "short_sleeve_center": self._lerp(shoulder, forearm, 0.32),
                "cuff_center": self._lerp(forearm, root, 0.68),
                "arm_rotation_z_deg": self._rotation_for_axis(shoulder, root),
                "bounds_cm": bounds((f'{side}Arm', f'{side}ForeArm')),
            }
        torso = {
            "chest_center": self._midpoint(center("leftShoulder"), center("rightShoulder")),
            "neck_center": self._safe_center(segmentation, vertices, ("neck", "head")),
            "neck_bounds_cm": bounds(('neck',)),
        }
        torso_points = [vertices[i] for name in ('hips', 'spine', 'spine1', 'spine2') for i in segmentation.get(name, [])]
        torso['bounds_cm'] = [min(p[a] for p in torso_points) for a in range(3)] + [max(p[a] for p in torso_points) for a in range(3)]
        from .wrist_placement import read_obj, wrist_frames
        _, faces = read_obj(body_assets_dir / f"{body_name}.obj")
        wrists = wrist_frames(vertices, faces, segmentation, segments)
        # Keep legacy shoulder transforms calibrated to their original landmarks;
        # the new cuff placement uses boundary centres, never the nearest vertex.
        return {"segments": segments, "torso": torso, "wrists": wrists,
                "height_cm": max(v[1] for v in vertices)-min(v[1] for v in vertices)}

    def _place_short_sleeves(
        self,
        spec: dict[str, Any],
        landmarks: dict[str, Any],
        relation: dict[str, Any],
        clearance: float,
    ) -> None:
        updates: dict[str, tuple[list[float], list[float]]] = {}
        for side, front, back, sign in (
            ("left", "short_sleeve_front", "short_sleeve_back", 1.0),
            ("right", "short_sleeve_front_right", "short_sleeve_back_right", -1.0),
        ):
            segment = landmarks["segments"][side]
            center = list(segment["short_sleeve_center"])
            center[0] += sign * clearance
            z_rotation = float(segment["arm_rotation_z_deg"])
            updates[front] = ([round(center[0], 6), round(center[1], 6), 30.0], [0.0, 0.0, round(z_rotation, 6)])
            updates[back] = ([round(center[0], 6), round(center[1], 6), -30.21], [0.0, 180.0, round(z_rotation, 6)])
        self._apply(spec, updates, relation)

    def _place_long_sleeves(
        self,
        spec: dict[str, Any],
        landmarks: dict[str, Any],
        relation: dict[str, Any],
        clearance: float,
    ) -> None:
        updates: dict[str, tuple[list[float], list[float]]] = {}
        for side, front, back, sign in (
            ("left", "long_sleeve_front", "long_sleeve_back", 1.0),
            ("right", "long_sleeve_front_right", "long_sleeve_back_right", -1.0),
        ):
            segment = landmarks["segments"][side]
            center = list(segment["long_sleeve_center"])
            center[0] += sign * clearance
            axis_offset = self._arm_axis_offset(segment, -LONG_CUFF_PROXIMAL_OFFSET_CM)
            center[0] += axis_offset[0]
            center[1] += axis_offset[1]
            z_rotation = float(segment["arm_rotation_z_deg"])
            updates[front] = ([round(center[0], 6), round(center[1], 6), LONG_SLEEVE_FRONT_Z], [0.0, 0.0, round(z_rotation, 6)])
            updates[back] = ([round(center[0], 6), round(center[1], 6), LONG_SLEEVE_BACK_Z], [0.0, 180.0, round(z_rotation, 6)])
            relation.setdefault("arm_height_placement", {})[side] = self._height_debug("long_sleeve")
            relation.setdefault("long_sleeve_axis_offset", {})[side] = {
                "strategy": "move_long_sleeve_by_previous_cuff_proximal_offset",
                "axis_offset_cm": LONG_CUFF_PROXIMAL_OFFSET_CM,
                "axis_offset_xy": [round(axis_offset[0], 6), round(axis_offset[1], 6)],
            }
        self._apply(spec, updates, relation)

    def _place_cuffs(self, spec: dict[str, Any], landmarks: dict[str, Any], relation: dict[str, Any], clearance: float) -> None:
        updates: dict[str, tuple[list[float], list[float]]] = {}
        for side, sleeve_front, sleeve_back, cuff_front, cuff_back in (
            ("left", "long_sleeve_front", "long_sleeve_back", "cuff_front", "cuff_back"),
            ("right", "long_sleeve_front_right", "long_sleeve_back_right", "cuff_front_right", "cuff_back_right"),
        ):
            segment = landmarks["segments"][side]
            z_rotation = float(segment["arm_rotation_z_deg"])
            front_rotation = [0.0, 0.0, round(z_rotation, 6)]
            back_rotation = [0.0, 180.0, round(z_rotation, 6)]
            axis_offset = [0.0, 0.0]
            front_translation, front_debug = self._translation_for_seam_center(
                spec, sleeve_front, cuff_front, front_rotation, LONG_CUFF_FRONT_Z, axis_offset
            )
            back_translation, back_debug = self._translation_for_seam_center(
                spec, sleeve_back, cuff_back, back_rotation, LONG_CUFF_BACK_Z, axis_offset
            )
            updates[cuff_front] = (front_translation, front_rotation)
            updates[cuff_back] = (back_translation, back_rotation)
            relation.setdefault("arm_height_placement", {})[side] = self._height_debug("cuff")
            relation.setdefault("cuff_seam_alignment", {})[cuff_front] = front_debug
            relation.setdefault("cuff_seam_alignment", {})[cuff_back] = back_debug
        self._apply(spec, updates, relation)

    def _translation_for_seam_center(
        self,
        spec: dict[str, Any],
        sleeve_panel: str,
        cuff_panel: str,
        cuff_rotation: list[float],
        cuff_z: float,
        axis_offset: list[float],
    ) -> tuple[list[float], dict[str, Any]]:
        sleeve_ref, cuff_ref = self._stitch_pair(spec, sleeve_panel, cuff_panel)
        panels = spec["pattern"]["panels"]
        sleeve_mid = self._edge_world_midpoint(panels, sleeve_ref)
        cuff_local_mid = self._edge_local_midpoint(panels[cuff_panel], cuff_ref)
        if sleeve_mid is None or cuff_local_mid is None:
            raise ValueError(f"failed to resolve cuff seam midpoint for {sleeve_panel} -> {cuff_panel}")
        rotated = self._rotate_local_xy(cuff_local_mid, cuff_rotation)
        translation = [
            round(sleeve_mid[0] - rotated[0] + axis_offset[0], 6),
            round(sleeve_mid[1] - rotated[1] + axis_offset[1], 6),
            cuff_z,
        ]
        debug = {
            "strategy": "align_cuff_top_join_center_to_moved_sleeve_cuff_join_center",
            "sleeve_panel": sleeve_panel,
            "cuff_panel": cuff_panel,
            "sleeve_edge": sleeve_ref["edge"],
            "cuff_edge": cuff_ref["edge"],
            "target_sleeve_edge_midpoint": [round(sleeve_mid[0], 6), round(sleeve_mid[1], 6), round(sleeve_mid[2], 6)],
            "cuff_local_edge_midpoint": [round(cuff_local_mid[0], 6), round(cuff_local_mid[1], 6)],
            "axis_offset_cm": round(math.hypot(axis_offset[0], axis_offset[1]), 6),
            "axis_offset_xy": [round(axis_offset[0], 6), round(axis_offset[1], 6)],
            "translation": translation,
        }
        return translation, debug

    def _arm_axis_offset(self, segment: dict[str, Any], distance: float) -> list[float]:
        shoulder = segment["shoulder"]
        hand_root = segment["hand_root"]
        vector_x = float(hand_root[0]) - float(shoulder[0])
        vector_y = float(hand_root[1]) - float(shoulder[1])
        length = math.hypot(vector_x, vector_y)
        if length <= 1.0e-8:
            return [0.0, 0.0]
        return [vector_x / length * distance, vector_y / length * distance]

    def _stitch_pair(self, spec: dict[str, Any], sleeve_panel: str, cuff_panel: str) -> tuple[dict[str, Any], dict[str, Any]]:
        for stitch in spec.get("pattern", {}).get("stitches", []):
            if not isinstance(stitch, list) or len(stitch) < 2:
                continue
            first, second = stitch[0], stitch[1]
            if not isinstance(first, dict) or not isinstance(second, dict):
                continue
            if first.get("panel") == sleeve_panel and second.get("panel") == cuff_panel:
                return first, second
            if first.get("panel") == cuff_panel and second.get("panel") == sleeve_panel:
                return second, first
        raise ValueError(f"no stitch found between {sleeve_panel} and {cuff_panel}")

    def _height_debug(self, _piece: str) -> dict[str, Any]:
        return {
            "strategy": "z_height_layering",
            "affected_pieces": ["long_sleeve", "cuff"],
            "long_sleeve_front_z": LONG_SLEEVE_FRONT_Z,
            "long_sleeve_back_z": LONG_SLEEVE_BACK_Z,
            "cuff_front_z": LONG_CUFF_FRONT_Z,
            "cuff_back_z": LONG_CUFF_BACK_Z,
            "front_z_gap_cm": abs(LONG_SLEEVE_FRONT_Z - LONG_CUFF_FRONT_Z),
            "back_z_gap_cm": abs(LONG_SLEEVE_BACK_Z - LONG_CUFF_BACK_Z),
        }

    def _debug_summary(self, spec: dict[str, Any]) -> dict[str, Any]:
        panels = spec.get("pattern", {}).get("panels", {})
        stitches = spec.get("pattern", {}).get("stitches", [])
        seams = self._seam_gap_summary(panels, stitches)
        overlaps = self._piece_overlap_summary(panels, stitches)
        return {
            "panel_count": len(panels),
            "stitch_count": len(stitches),
            "seam_gap_cm": seams,
            "piece_overlap_pairs": overlaps,
            "body_penetration": {"status": "not_computed_pre_boxmesh", "max_depth_cm": None},
        }

    def _seam_gap_summary(self, panels: dict[str, Any], stitches: list[Any]) -> dict[str, Any]:
        gaps: list[float] = []
        for stitch in stitches:
            if not isinstance(stitch, list) or len(stitch) < 2:
                continue
            first = self._edge_world_midpoint(panels, stitch[0])
            second = self._edge_world_midpoint(panels, stitch[1])
            if first is None or second is None:
                continue
            gaps.append(self._dist3(first, second))
        if not gaps:
            return {"count": 0, "mean": None, "max": None}
        return {"count": len(gaps), "mean": sum(gaps) / len(gaps), "max": max(gaps)}

    def _piece_overlap_summary(self, panels: dict[str, Any], stitches: list[Any]) -> dict[str, Any]:
        connected = set()
        for stitch in stitches:
            if not isinstance(stitch, list) or len(stitch) < 2:
                continue
            edge_a, edge_b = stitch[0], stitch[1]
            if isinstance(edge_a, dict) and isinstance(edge_b, dict):
                connected.add(tuple(sorted((edge_a.get("panel"), edge_b.get("panel")))))
        boxes = {name: self._world_bbox(panel) for name, panel in panels.items()}
        pairs: list[dict[str, Any]] = []
        names = list(boxes)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if tuple(sorted((left, right))) in connected:
                    continue
                overlap = self._xy_overlap_area(boxes[left], boxes[right])
                if overlap > 1.0:
                    pairs.append({"panels": [left, right], "xy_overlap_area": overlap})
        return {"count": len(pairs), "largest": pairs[:10]}

    def _edge_world_midpoint(self, panels: dict[str, Any], edge_ref: dict[str, Any]) -> list[float] | None:
        panel = panels.get(edge_ref.get("panel"))
        if not panel:
            return None
        local = self._edge_local_midpoint(panel, edge_ref)
        if local is None:
            return None
        return self._world_point(panel, local)

    def _edge_local_midpoint(self, panel: dict[str, Any], edge_ref: dict[str, Any]) -> list[float] | None:
        edge_index = edge_ref.get("edge")
        edges = panel.get("edges", [])
        vertices = panel.get("vertices", [])
        if not isinstance(edge_index, int) or edge_index >= len(edges):
            return None
        edge = edges[edge_index]
        try:
            a = vertices[edge["endpoints"][0]]
            b = vertices[edge["endpoints"][1]]
        except (KeyError, IndexError, TypeError):
            return None
        return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]

    def _rotate_local_xy(self, point: list[float], rotation: list[float]) -> list[float]:
        y_angle = math.radians(float(rotation[1] if len(rotation) > 1 else 0.0))
        z_angle = math.radians(float(rotation[2] if len(rotation) > 2 else 0.0))
        x = float(point[0]) * math.cos(y_angle)
        y = float(point[1])
        return [
            x * math.cos(z_angle) - y * math.sin(z_angle),
            x * math.sin(z_angle) + y * math.cos(z_angle),
        ]

    def _world_bbox(self, panel: dict[str, Any]) -> tuple[float, float, float, float]:
        points = [self._world_point(panel, vertex) for vertex in panel.get("vertices", [])]
        if not points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _world_point(self, panel: dict[str, Any], point: list[float]) -> list[float]:
        tx, ty, tz = panel.get("translation", [0.0, 0.0, 0.0])
        rotation = panel.get("rotation", [0.0, 0.0, 0.0])
        x, y = self._rotate_local_xy(point, rotation)
        return [x + float(tx), y + float(ty), float(tz)]

    def _xy_overlap_area(self, left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        return width * height

    def _safe_center(self, segmentation: dict[str, Any], vertices: list[list[float]], names: tuple[str, ...]) -> list[float] | None:
        for name in names:
            if name not in segmentation:
                continue
            points = [vertices[index] for index in segmentation[name] if index < len(vertices)]
            if points:
                return [round(sum(point[axis] for point in points) / len(points), 6) for axis in range(3)]
        return None

    def _load_obj_vertices(self, path: Path) -> list[list[float]]:
        vertices: list[list[float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "v":
                vertices.append([float(parts[1]) * 100.0, float(parts[2]) * 100.0, float(parts[3]) * 100.0])
        if not vertices:
            raise ValueError(f"no vertices found in {path}")
        return vertices

    def _apply(self, spec: dict[str, Any], updates: dict[str, tuple[list[float], list[float]]], relation: dict[str, Any]) -> None:
        panels = spec["pattern"]["panels"]
        for name, (translation, rotation) in updates.items():
            if name not in panels:
                continue
            before = {"translation": panels[name].get("translation"), "rotation": panels[name].get("rotation")}
            panels[name]["translation"] = translation
            panels[name]["rotation"] = rotation
            relation["piece_placement"][name] = {"before": before, "after": {"translation": translation, "rotation": rotation}}

    def _midpoint(self, a: list[float], b: list[float]) -> list[float]:
        return self._lerp(a, b, 0.5)

    def _lerp(self, a: list[float], b: list[float], t: float) -> list[float]:
        return [round(float(a[index]) + (float(b[index]) - float(a[index])) * t, 6) for index in range(3)]

    def _rotation_for_axis(self, start: list[float], end: list[float]) -> float:
        vector_x = float(end[0]) - float(start[0])
        vector_y = float(end[1]) - float(start[1])
        sign = 1.0 if vector_x >= 0.0 else -1.0
        return sign * math.degrees(math.atan2(abs(vector_x), max(abs(vector_y), 1.0e-8)))

    def _dist3(self, a: list[float], b: list[float]) -> float:
        return sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)) ** 0.5


TposePlacement = BodyPosePlacement
