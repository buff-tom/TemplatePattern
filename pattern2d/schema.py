from __future__ import annotations

from typing import Any


SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

PIECE_ORDER = {
    "long_sleeve": [
        "front_body_right",
        "front_body_left",
        "back_body",
        "back_yoke",
        "collar_stand",
        "collar_fall",
        "long_sleeve",
        "cuff",
        "placket_tip",
        "placket_strip",
    ],
    "short_sleeve": [
        "front_body_right",
        "front_body_left",
        "back_body",
        "back_yoke",
        "collar_stand",
        "collar_fall",
        "short_sleeve",
    ],
}

BOUNDARY_ORDERS = {
    "front_body_right": ["neckline", "shoulder", "armhole", "side_seam", "hem", "cf"],
    "front_body_left": ["neckline", "shoulder", "armhole", "side_seam", "hem", "cf"],
    "back_body": ["yoke_join", "armhole_left_part", "side_seam_left", "back_hem", "side_seam_right", "armhole_right_part"],
    "back_yoke": ["back_neckline", "shoulder_left", "armhole_left_part", "yoke_join", "armhole_right_part", "shoulder_right"],
    "collar_stand": ["collar_stand_top", "collar_stand_bottom"],
    "collar_fall": ["collar_fall_top", "collar_fall_right", "collar_fall_bottom", "collar_fall_left"],
    "short_sleeve": ["sleeve_cap", "underarm_back", "cuff", "underarm_front"],
    "long_sleeve": ["sleeve_cap", "underarm_back", "sleeve_cuff_join", "underarm_front"],
    "cuff": ["cuff_top_join", "cuff_short_end", "cuff_bottom_edge"],
    "placket_tip": ["placket_tip_top", "placket_tip_side", "placket_tip_bottom"],
    "placket_strip": ["placket_strip_top", "placket_strip_side", "placket_strip_bottom"],
}

PLACEMENT_RULES = {
    "front_body_right": {"region": "front_torso", "side": "right", "offset": 30},
    "front_body_left": {"region": "front_torso", "side": "left", "offset": 30},
    "back_body": {"region": "back_torso", "offset": 30},
    "back_yoke": {"region": "upper_back", "offset": 28},
    "collar_stand": {"region": "neck", "layer": "stand", "offset": 12},
    "collar_fall": {"region": "neck", "layer": "fall", "offset": 14},
    "short_sleeve": {"region": "arm_pair", "sleeve": "short", "offset": 20},
    "long_sleeve": {"region": "arm_pair", "sleeve": "long", "offset": 20},
    "cuff": {"region": "wrist_pair", "offset": 10},
    "placket_tip": {"region": "sleeve_placket", "offset": 6},
    "placket_strip": {"region": "sleeve_placket", "offset": 6},
}


def seam(a_piece: str, a_edge: str, b_piece: str, b_edge: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "a": {"piece": a_piece, "edge": a_edge},
        "b": {"piece": b_piece, "edge": b_edge},
        "direction": extra.pop("direction", "same"),
        "target": extra.pop("target", "sew"),
    }
    item.update(extra)
    return item


SHORT_SEAMS = [
    seam("front_body_left", "side_seam", "back_body", "side_seam_left", direction="opposite"),
    seam("front_body_right", "side_seam", "back_body", "side_seam_right", direction="opposite"),
    seam("front_body_left", "shoulder", "back_yoke", "shoulder_left", direction="opposite"),
    seam("front_body_right", "shoulder", "back_yoke", "shoulder_right", direction="opposite"),
    seam("back_body", "yoke_join", "back_yoke", "yoke_join", direction="opposite"),
    seam("collar_stand", "collar_stand_top", "collar_fall", "collar_fall_bottom"),
    seam("collar_stand", "collar_stand_bottom", "front_body_left", "neckline"),
    seam("collar_stand", "collar_stand_bottom", "back_yoke", "back_neckline"),
    seam("collar_stand", "collar_stand_bottom", "front_body_right", "neckline"),
    seam("short_sleeve", "sleeve_cap", "front_body_left", "armhole", direction="opposite", mirrored_direction="same"),
    seam("short_sleeve", "sleeve_cap", "back_yoke", "armhole_left_part", direction="opposite", mirrored_direction="same"),
    seam("short_sleeve", "sleeve_cap", "back_body", "armhole_left_part", direction="opposite", mirrored_direction="same"),
    seam("short_sleeve", "underarm_front", "short_sleeve", "underarm_back", direction="opposite", mirrored_direction="opposite"),
]

LONG_SEAMS = [
    seam("front_body_left", "side_seam", "back_body", "side_seam_left", direction="opposite"),
    seam("front_body_right", "side_seam", "back_body", "side_seam_right", direction="opposite"),
    seam("front_body_left", "shoulder", "back_yoke", "shoulder_left", direction="opposite"),
    seam("front_body_right", "shoulder", "back_yoke", "shoulder_right", direction="opposite"),
    seam("back_body", "yoke_join", "back_yoke", "yoke_join"),
    seam("collar_stand", "collar_stand_top", "collar_fall", "collar_fall_bottom"),
    seam("collar_stand", "collar_stand_bottom", "front_body_left", "neckline"),
    seam("collar_stand", "collar_stand_bottom", "back_yoke", "back_neckline"),
    seam("collar_stand", "collar_stand_bottom", "front_body_right", "neckline"),
    seam("long_sleeve", "sleeve_cap", "front_body_left", "armhole", direction="same", mirrored_direction="same"),
    seam("long_sleeve", "sleeve_cap", "back_yoke", "armhole_left_part", direction="same", mirrored_direction="same"),
    seam("long_sleeve", "sleeve_cap", "back_body", "armhole_left_part", direction="opposite", mirrored_direction="opposite"),
    seam("long_sleeve", "underarm_front", "long_sleeve", "underarm_back", direction="opposite", mirrored_direction="opposite"),
    seam("long_sleeve", "sleeve_cuff_join", "cuff", "cuff_top_join", direction="opposite", mirrored_direction="opposite"),
    seam("placket_strip", "placket_strip_side", "long_sleeve", "underarm_back", direction="opposite", target="attach"),
    seam("placket_tip", "placket_tip_bottom", "placket_strip", "placket_strip_top", direction="opposite", target="attach"),
]
