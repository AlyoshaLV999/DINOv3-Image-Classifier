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
    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    try:
        accepted, rejected, active_csvs = run_inference(
            project_root=PROJECT_ROOT,
            dataset=args.dataset,
            input_dir=input_dir,
            probability_threshold=args.threshold,
        )
    except ProjectError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"accepted={accepted} rejected={rejected}")
    for active_csv in active_csvs:
        print(f"active_learning_csv={active_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
