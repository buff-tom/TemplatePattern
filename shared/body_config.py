from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

from .io import read_json


REQUIRED_BODY_KEYS = ("height", "chest", "waist", "hip", "shoulder_width", "arm_length")


@dataclass(frozen=True)
class BodyConfig:
    sample_name: str
    body_input: dict[str, float]

    @classmethod
    def load(cls, path: str | Path) -> "BodyConfig":
        data = read_json(path)
        body = data.get("body_input", data)
        missing = [key for key in REQUIRED_BODY_KEYS if key not in body]
        if missing:
            raise ValueError(f"body config missing keys: {', '.join(missing)}")
        clean = {key: float(body[key]) for key in REQUIRED_BODY_KEYS}
        invalid = [key for key, value in clean.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"body config requires positive finite measurements in mm: {', '.join(invalid)}")
        name = str(data.get("sample_name") or "body_input")
        return cls(name, clean)

    def size_distance(self, reference: dict[str, Any]) -> float:
        weights = {"height": 0.8, "chest": 1.5, "waist": 1.0, "hip": 1.0, "shoulder_width": 1.2, "arm_length": 1.0}
        total = 0.0
        for key, weight in weights.items():
            scale = max(abs(float(reference.get(key, 1.0))), 1.0)
            total += weight * ((self.body_input[key] - float(reference[key])) / scale) ** 2
        return total ** 0.5
