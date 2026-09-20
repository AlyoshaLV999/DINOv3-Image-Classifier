"""Shared deterministic, filesystem, logging, and serialization helpers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


CODE_VERSION = "1.0.0"
CHECKPOINT_FORMAT_VERSION = 1
ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".heic", ".webp"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ProjectError(RuntimeError):
    """An actionable runtime error intended for an end user."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, stable_json_dumps(value) + "\n")


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with temp_path.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise



def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with source.open("rb") as source_handle, temp_path.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise



def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("image_auto_classifier")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ProjectError("CUDA GPU is required; no CUDA device is available.")
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise ProjectError("Checkpoint RNG state must be a mapping")

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ProjectError(f"Checkpoint RNG state is incomplete: {sorted(missing)}")

    cuda_available = torch.cuda.is_available()
    try:
        cpu_state = state["torch_cpu"]
        if not isinstance(cpu_state, torch.Tensor):
            raise TypeError("CPU RNG state must be a torch.Tensor")
        cpu_state = cpu_state.detach().to(device="cpu").contiguous()

        normalized_cuda_states: list[torch.Tensor] = []
        if cuda_available:
            cuda_states = state["torch_cuda"]
            if not isinstance(cuda_states, (list, tuple)):
                raise TypeError("CUDA RNG state must be a sequence of torch.Tensor values")
            for index, cuda_state in enumerate(cuda_states):
                if not isinstance(cuda_state, torch.Tensor):
                    raise TypeError(f"CUDA RNG state at index {index} must be a torch.Tensor")
                normalized_cuda_states.append(cuda_state.detach().to(device="cpu").contiguous())

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(cpu_state)
        if cuda_available:
            torch.cuda.set_rng_state_all(normalized_cuda_states)
    except (IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ProjectError(f"Checkpoint RNG state cannot be restored: {error}") from error


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def is_allowed_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES


@contextlib.contextmanager
def temporary_state_dict(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> Iterator[None]:
    original = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
    module.load_state_dict(state, strict=True)
    try:
        yield
    finally:
        module.load_state_dict(original, strict=True)
