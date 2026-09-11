from __future__ import annotations

import argparse
import json

from TemplatePattern_final.pattern2d.short_pipeline import ShortSleevePatternPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final short-sleeve 2D pattern from body config.")
    parser.add_argument("--body-config", default="config/body_short.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    outputs = ShortSleevePatternPipeline(args.body_config, args.output_dir).run()
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
