"""Strict checkpoint schema, atomic persistence, and model registration."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from .common import (
    CHECKPOINT_FORMAT_VERSION,
    CODE_VERSION,
    ProjectError,
    atomic_copy,
    atomic_torch_save,
    atomic_write_json,
)
from .config import AppConfig
from .model import MetricArcFaceModel, ParameterEMA


REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "code_version",
        "model_state",
        "lora_state",
        "arcface_state",
        "ema_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "epoch",
        "best_score",
        "rng_state",
        "tag_to_id",
        "split_manifest_hash",
        "config",
        "inference",
        "history",
        "monitor",
    }
)


REGISTRY_POINTER_NAME = "current.json"
REGISTRY_VERSIONS_DIRECTORY = "versions"
REGISTRY_FORMAT_VERSION = 1
_REGISTRATION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")

_logger = logging.getLogger(__name__)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise ProjectError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as error:
        raise ProjectError(f"Cannot read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProjectError("Checkpoint payload must be a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise ProjectError(f"Checkpoint is not resumable; missing keys: {sorted(missing)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ProjectError("Checkpoint format version is not supported by this project")
    if not isinstance(payload["tag_to_id"], Mapping):
        raise ProjectError("Checkpoint tag_to_id is invalid")
    return payload


def latest_epoch_checkpoint(models_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in models_dir.glob("epoch_*.ckpt"):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        candidates.append((epoch, path))
    return max(candidates, default=(0, None), key=lambda value: value[0])[1]


def build_checkpoint(
    *,
    model: MetricArcFaceModel,
    ema: ParameterEMA,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Mapping[str, Any],
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    rng_state: Mapping[str, Any],
    tag_to_id: Mapping[str, int],
    split_manifest_hash: str,
    config: AppConfig,
    inference: Mapping[str, Any],
    history: Mapping[str, Any],
    monitor: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = model.state_dict()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "code_version": CODE_VERSION,
        "model_state": model_state,
        "lora_state": {name: value for name, value in model_state.items() if "lora_" in name},
        "arcface_state": {"weight": model.arcface.weight.detach().cpu()},
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": dict(scheduler_state),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_score": float(best_score),
        "rng_state": dict(rng_state),
        "tag_to_id": dict(tag_to_id),
        "split_manifest_hash": split_manifest_hash,
        "config": config.to_dict(),
        "inference": dict(inference),
        "history": dict(history),
        "monitor": dict(monitor),
    }


def save_epoch_checkpoint(models_dir: Path, epoch: int, checkpoint: Mapping[str, Any]) -> Path:
    path = models_dir / f"epoch_{epoch:03d}.ckpt"
    atomic_torch_save(path, checkpoint)
    epoch_paths = sorted(models_dir.glob("epoch_*.ckpt"), key=lambda item: item.name)
    for old_path in epoch_paths[:-3]:
        old_path.unlink()
    return path


def registered_model_paths(registry_dir: Path) -> tuple[Path, Path]:
    """Resolve the checkpoint and metadata selected by one atomic registry pointer."""
    pointer_path = registry_dir / REGISTRY_POINTER_NAME
    if not pointer_path.is_file():
        raise ProjectError(
            f"Registered model pointer not found: {pointer_path}. Train a model before running inference."
        )
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ProjectError(f"Cannot read registered model pointer {pointer_path}: {error}") from error
    if not isinstance(pointer, Mapping):
        raise ProjectError("Registered model pointer must be a JSON object")
    if set(pointer) != {"format_version", "version"}:
        raise ProjectError("Registered model pointer has an unsupported schema")
    if pointer["format_version"] != REGISTRY_FORMAT_VERSION:
        raise ProjectError("Registered model pointer format is not supported by this project")
    version = pointer["version"]
    if not isinstance(version, str) or not _REGISTRATION_ID_PATTERN.fullmatch(version):
        raise ProjectError("Registered model pointer contains an invalid version identifier")

    version_dir = registry_dir / REGISTRY_VERSIONS_DIRECTORY / version
    checkpoint_path = version_dir / "best.ckpt"
    metadata_path = version_dir / "metadata.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise ProjectError(
            "Registered model pointer references an incomplete version; rerun training to publish a new model."
        )
    return checkpoint_path, metadata_path


def _metadata(
    *,
    dataset: str,
    task_name: str,
    tag_to_id: Mapping[str, int],
    config: AppConfig,
    inference: Mapping[str, Any],
    epoch: int,
    split_manifest_hash: str,
    registry_version: str,
) -> dict[str, Any]:
    id_to_tag = [tag for tag, _ in sorted(tag_to_id.items(), key=lambda pair: pair[1])]
    gates = inference["similarity_thresholds"]
    if isinstance(gates, torch.Tensor):
        gate_values = gates.cpu().tolist()
    else:
        gate_values = list(gates)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "code_version": CODE_VERSION,
        "dataset": dataset,
        "task_name": task_name,
        "epoch": epoch,
        "registry_version": registry_version,
        "tag_to_id": dict(tag_to_id),
        "tags": id_to_tag,
        "probability_threshold": float(inference["probability_threshold"]),
        "temperature": float(inference["temperature"]),
        "similarity_thresholds": {tag: float(gate_values[class_id]) for tag, class_id in tag_to_id.items()},
        "preprocessing": {
            "input_size": config.data.input_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "cache_long_side": config.data.cache_long_side,
        },
        "model": config.model.__dict__,
        "split_manifest_hash": split_manifest_hash,
    }


def prune_stale_registry_versions(registry_dir: Path) -> None:
    """Delete every registry version directory that the active pointer does not reference.

    Must be called only after the pointer has been atomically switched to the newest
    version, so a crash during pruning can at worst leave stale directories behind; it
    can never remove the version that inference resolves through ``current.json``.
    The pointer is re-read here rather than trusting the caller's in-memory version,
    which keeps pruning safe if the pointer was updated concurrently. Stale-version
    removal is housekeeping only, therefore every failure is logged as a warning and
    never aborts training (e.g. files briefly locked by a concurrent reader).

    :param registry_dir: Registry root directory, i.e. ``models/<dataset>``.
    """
    pointer_path = registry_dir / REGISTRY_POINTER_NAME
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _logger.warning(
            "Skip pruning stale registry versions; cannot read pointer %s: %s", pointer_path, error
        )
        return
    active_version = pointer.get("version") if isinstance(pointer, Mapping) else None
    if not isinstance(active_version, str) or not _REGISTRATION_ID_PATTERN.fullmatch(active_version):
        _logger.warning("Skip pruning stale registry versions; pointer %s has an invalid schema", pointer_path)
        return
    versions_dir = registry_dir / REGISTRY_VERSIONS_DIRECTORY
    if not versions_dir.is_dir():
        return
    try:
        entries = sorted(versions_dir.iterdir(), key=lambda path: path.name)
    except OSError as error:
        _logger.warning("Cannot list registry versions in %s: %s", versions_dir, error)
        return
    removed = 0
    for version_path in entries:
        if not version_path.is_dir() or version_path.name == active_version:
            continue
        try:
            shutil.rmtree(version_path)
        except OSError as error:
            _logger.warning("Cannot remove stale registry version %s: %s", version_path, error)
            continue
        removed += 1
    if removed:
        _logger.info("Pruned %d stale registry version(s); retained %s", removed, active_version)


def register_best_model(
    *,
    epoch_checkpoint: Path,
    project_root: Path,
    dataset: str,
    task_name: str,
    tag_to_id: Mapping[str, int],
    config: AppConfig,
    inference: Mapping[str, Any],
    epoch: int,
    split_manifest_hash: str,
    task_models_dir: Path,
) -> None:
    """Publish one registry version, atomically switch the pointer, then keep only the newest best.

    Every time validation improves, an immutable version (``versions/<id>/best.ckpt``
    plus ``metadata.json``) is published and ``current.json`` is atomically repointed at
    it, so concurrent inference never observes a half-written model. Afterwards every
    version other than the newly active one is deleted, keeping exactly one best model
    on disk; historical rollback points are intentionally not retained.
    """
    task_best = task_models_dir / "best.ckpt"
    atomic_copy(epoch_checkpoint, task_best)

    registry_dir = project_root / "models" / dataset
    registry_version = f"epoch_{epoch:03d}_{uuid.uuid4().hex}"
    version_dir = registry_dir / REGISTRY_VERSIONS_DIRECTORY / registry_version
    atomic_copy(task_best, version_dir / "best.ckpt")
    atomic_write_json(
        version_dir / "metadata.json",
        _metadata(
            dataset=dataset,
            task_name=task_name,
            tag_to_id=tag_to_id,
            config=config,
            inference=inference,
            epoch=epoch,
            split_manifest_hash=split_manifest_hash,
            registry_version=registry_version,
        ),
    )
    atomic_write_json(
        registry_dir / REGISTRY_POINTER_NAME,
        {
            "format_version": REGISTRY_FORMAT_VERSION,
            "version": registry_version,
        },
    )
    prune_stale_registry_versions(registry_dir)


def validate_metadata(
    path: Path,
    checkpoint: Mapping[str, Any],
    dataset: str,
    registry_version: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ProjectError(f"Model metadata not found: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ProjectError(f"Cannot read metadata {path}: {error}") from error
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION or metadata.get("dataset") != dataset:
        raise ProjectError("Model metadata is incompatible with the registered checkpoint")
    if metadata.get("registry_version") != registry_version:
        raise ProjectError("Model metadata does not match the registered model pointer")
    if metadata.get("tag_to_id") != checkpoint["tag_to_id"]:
        raise ProjectError("Model metadata and checkpoint tag mappings do not match")
    if metadata.get("split_manifest_hash") != checkpoint["split_manifest_hash"]:
        raise ProjectError("Model metadata and checkpoint manifests do not match")
    return metadata