from __future__ import annotations

from pathlib import Path

from TemplatePattern_final.shared.io import ROOT

from .base_pipeline import TemplatePatternPipeline


class LongSleevePatternPipeline(TemplatePatternPipeline):
    style = "long_sleeve"
    garment_type = "shirt_long_sleeve"

    def template_path(self, size: str) -> Path:
        return ROOT / "templates" / "long_sleeve" / f"long_sleeve_shirt_template_{size}.json"

