from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from .io import read_json


SCHEMA_VERSION = 1
STYLES = ("long_sleeve", "short_sleeve")


@dataclass(frozen=True)
class Stage1Bundle:
    root: Path
    manifest: dict[str, Any]
    request: dict[str, Any]
    pattern: Path
    body_target: Path
    style: str
    sample_name: str

    @classmethod
    def load(cls, directory: str | Path) -> "Stage1Bundle":
        root = Path(directory).resolve()
        manifest_path = root / "stage1_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage1 manifest is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage1 schema_version {manifest.get('schema_version')!r} in {manifest_path}; "
                f"expected {SCHEMA_VERSION}"
            )
        if manifest.get("status") != "completed":
            raise ValueError(f"Stage1 task is not completed: {manifest_path}")
        outputs = manifest.get("outputs") or {}
        cls._resolve_member(root, outputs.get("svg"), "svg")
        request_path = cls._resolve_member(root, outputs.get("stage2_request"), "stage2_request")
        request = read_json(request_path)
        if request.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported Stage2 request schema in {request_path}")
        style = str(request.get("style") or "")
        if style not in STYLES:
            raise ValueError(f"invalid style {style!r} in {request_path}")
        pattern = cls._resolve_member(root, request.get("pattern"), "pattern")
        body_target = cls._resolve_member(root, request.get("body_target"), "body_target")
        return cls(
            root=root,
            manifest=manifest,
            request=request,
            pattern=pattern,
            body_target=body_target,
            style=style,
            sample_name=str(manifest.get("sample_name") or "body_input"),
        )

    def digest(self) -> str:
        digest = hashlib.sha256()
        for path in sorted((self.root / name for name in ("stage1_manifest.json", "pattern.json", "body_target.json", "stage2_request.json"))):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _resolve_member(root: Path, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Stage1 bundle does not declare {label}")
        member = (root / value).resolve()
        try:
            member.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Stage1 {label} escapes task directory: {value}") from exc
        if not member.is_file():
            raise FileNotFoundError(f"Stage1 {label} is missing: {member}")
        return member
