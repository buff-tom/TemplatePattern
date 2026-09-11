from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAX_LINES = 500
FORBIDDEN = ("TemplatePattern_" + "v1", "TemplatePattern_" + "v2", "/home" + "/user")


def main() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        line_count = len(text.splitlines())
        if line_count > MAX_LINES:
            failures.append(f"{path.relative_to(ROOT)} has {line_count} lines")
        for token in FORBIDDEN:
            if token in text:
                failures.append(f"{path.relative_to(ROOT)} contains forbidden token {token}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"ok: all Python files are <= {MAX_LINES} lines and do not import old TemplatePattern dirs")


if __name__ == "__main__":
    main()
