from __future__ import annotations

from pathlib import Path

from TemplatePattern_final.shared.io import ROOT

from .base_pipeline import TemplatePatternPipeline


class ShortSleevePatternPipeline(TemplatePatternPipeline):
    style = "short_sleeve"
    garment_type = "shirt_short_sleeve"

    def template_path(self, size: str) -> Path:
        return ROOT / "templates" / "short_sleeve" / f"short_sleeve_shirt_template_{size}.json"

