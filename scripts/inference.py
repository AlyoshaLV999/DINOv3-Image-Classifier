#!/usr/bin/env python3
"""Classify first-level input images with one or all registered dataset models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_auto_classifier.common import ProjectError  # noqa: E402
from image_auto_classifier.inferencer import run_inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference with registered DINOv3 image classifiers")
    parser.add_argument(
        "--dataset",
        default="__UNITEINFER__",
        help="Dataset registry name; omit or use __UNITEINFER__ to compare every registered dataset model",
    )
    parser.add_argument("--input", default="input", help="First-level input image directory")
    parser.add_argument("--threshold", type=float, help="Override calibrated probability threshold")
    parser.add_argument(
        "--config",
        type=Path,
        help="Shared inference settings YAML (defaults to configs/inference.yaml)",
    )
    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    try:
        accepted, rejected = run_inference(
            project_root=PROJECT_ROOT,
            dataset=args.dataset,
            input_dir=input_dir,
            probability_threshold=args.threshold,
            inference_config_path=(
                args.config if args.config is None or args.config.is_absolute()
                else PROJECT_ROOT / args.config
            ),
        )
    except (ProjectError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"accepted={accepted} rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
