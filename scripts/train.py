#!/usr/bin/env python3
"""Train a configuration-named task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_auto_classifier.common import ProjectError, configure_logging  # noqa: E402
from image_auto_classifier.config import ConfigurationError, load_config  # noqa: E402
from image_auto_classifier.trainer import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DINOv3 LoRA image classifier")
    parser.add_argument("--config", required=True, help="Path relative to the project root or an absolute YAML path")
    parser.add_argument("--resume", choices=["auto"], help="Resume only from this task's latest epoch checkpoint")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        config = load_config(config_path)
        task_name = config_path.stem
        logger = configure_logging(PROJECT_ROOT / "logs" / task_name / f"{task_name}.log")
        train(
            project_root=PROJECT_ROOT,
            task_name=task_name,
            config=config,
            resume_auto=args.resume == "auto",
            logger=logger,
        )
        return 0
    except (ProjectError, ConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
