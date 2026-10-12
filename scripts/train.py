#!/usr/bin/env python3
"""Train configured tasks or build image caches without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_auto_classifier.common import ProjectError, configure_logging  # noqa: E402
from image_auto_classifier.config import ConfigurationError, load_config  # noqa: E402
from image_auto_classifier.data import audit_dataset  # noqa: E402
from image_auto_classifier.trainer import train  # noqa: E402


DEFAULT_CACHE_LONG_SIDE = 1280
# ``configs/inference.yaml`` stores shared inference settings and uses a
# different schema from task configs, so auto-discovery must never parse it as
# an ``AppConfig``. Explicit ``--config`` arguments stay user-controlled and are
# still validated by ``load_config``.
INFERENCE_CONFIG_NAME = "inference.yaml"


def _task_config_paths() -> list[Path]:
    """Return every task YAML under ``configs/`` in deterministic order.

    Shared, non-task configuration files (currently ``inference.yaml``) are
    excluded because they use the inference schema and would otherwise be
    rejected by ``parse_config`` when tasks are auto-discovered.

    :return: Sorted task configuration paths; empty when the directory holds no
        task configs.
    """
    config_dir = PROJECT_ROOT / "configs"
    return sorted(
        (
            path
            for path in (*config_dir.glob("*.yaml"), *config_dir.glob("*.yml"))
            if path.name != INFERENCE_CONFIG_NAME
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _config_paths(config_arg: str | None) -> list[Path]:
    """Resolve an explicit config path or return every discovered task YAML."""
    if config_arg is not None:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        return [config_path]
    paths = _task_config_paths()
    if not paths:
        raise ProjectError(f"No YAML configuration files found in {PROJECT_ROOT / 'configs'}")
    return paths


def _cache_datasets(config_arg: str | None) -> list[tuple[str, int]]:
    """Return dataset names and cache sizes for cache-only mode."""
    if config_arg is not None:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = load_config(config_path)
        return [(config.dataset, config.data.cache_long_side)]

    datasets_root = PROJECT_ROOT / "datasets"
    if not datasets_root.is_dir():
        raise ProjectError(f"Datasets directory not found: {datasets_root}")
    datasets = sorted(
        (path.name for path in datasets_root.iterdir() if path.is_dir()),
        key=lambda name: (name.casefold(), name),
    )
    if not datasets:
        raise ProjectError(f"No dataset directories found in {datasets_root}")

    cache_sizes: dict[str, int] = {}
    for config_path in _task_config_paths():
        config = load_config(config_path)
        cache_size = config.data.cache_long_side
        previous_size = cache_sizes.get(config.dataset)
        if previous_size is not None and previous_size != cache_size:
            raise ConfigurationError(
                f"Configurations for dataset '{config.dataset}' specify conflicting cache_long_side values"
            )
        cache_sizes[config.dataset] = cache_size
    return [(dataset, cache_sizes.get(dataset, DEFAULT_CACHE_LONG_SIDE)) for dataset in datasets]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DINOv3 LoRA image classifier or generate image caches")
    parser.add_argument("--config", help="YAML path; defaults to all configs (or datasets with --cache_gen)")
    parser.add_argument("--resume", choices=["auto"], help="Resume from the latest checkpoint when available")
    parser.add_argument("--cache_gen", action="store_true", help="Generate cache files without training")
    args = parser.parse_args()
    try:
        if args.cache_gen:
            if args.resume is not None:
                parser.error("--resume cannot be used with --cache_gen")
            failures: list[str] = []
            for dataset, cache_long_side in _cache_datasets(args.config):
                logger = configure_logging(PROJECT_ROOT / "logs" / "cache_gen" / f"{dataset}.log")
                try:
                    audit_dataset(
                        project_root=PROJECT_ROOT,
                        dataset=dataset,
                        task_log_dir=PROJECT_ROOT / "logs" / "cache_gen" / dataset,
                        cache_long_side=cache_long_side,
                        logger=logger,
                        create_split_manifest=False,
                    )
                except (ProjectError, OSError) as error:
                    failures.append(f"{dataset}: {error}")
                    logger.exception("Cache generation failed for dataset %s", dataset)
            if failures:
                for failure in failures:
                    print(f"ERROR: {failure}", file=sys.stderr)
                return 2
            return 0

        failures = []
        for config_path in _config_paths(args.config):
            task_name = config_path.stem
            logger = None
            try:
                config = load_config(config_path)
                logger = configure_logging(PROJECT_ROOT / "logs" / task_name / f"{task_name}.log")
                train(
                    project_root=PROJECT_ROOT,
                    task_name=task_name,
                    config=config,
                    resume_auto=args.resume == "auto",
                    logger=logger,
                )
            except (ProjectError, ConfigurationError, OSError) as error:
                failures.append(f"{config_path.name}: {error}")
                if logger is not None:
                    logger.exception("Training failed for configuration %s", config_path)
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 2
        return 0
    except (ProjectError, ConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())