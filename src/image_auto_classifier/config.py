"""Configuration parsing and strict validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


LOCKED_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
LOCKED_LORA_RANK = 16
LOCKED_LORA_ALPHA = 32
LOCKED_MIXED_PRECISION = True


class ConfigurationError(ValueError):
    """Raised when a task configuration is malformed or unsafe."""


@dataclass(frozen=True)
class DataConfig:
    cache_long_side: int = 1280
    input_size: int = 256
    num_workers: int = 2


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 50
    micro_batch: int = 8
    grad_accum: int = 4
    mixed_precision: bool = LOCKED_MIXED_PRECISION
    early_stop_patience: int = 10
    adaptation_epochs: int = 12



@dataclass(frozen=True)
class ModelConfig:
    name: str = LOCKED_MODEL_NAME
    lora_rank: int = LOCKED_LORA_RANK
    lora_alpha: int = LOCKED_LORA_ALPHA


@dataclass(frozen=True)
class AppConfig:
    dataset: str
    seed: int = 3407
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{name}' must be a mapping")
    return value


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown keys in '{name}': {', '.join(sorted(unknown))}")


def parse_config(raw: Mapping[str, Any]) -> AppConfig:
    """Create a validated immutable config from a YAML mapping or checkpoint mapping."""
    _reject_unknown(raw, {"dataset", "seed", "data", "train", "model"}, "root")
    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ConfigurationError("'dataset' must be a non-empty string")
    dataset_path = Path(dataset)
    if dataset_path.name != dataset or dataset in {".", ".."} or "/" in dataset or "\\" in dataset:
        raise ConfigurationError("'dataset' must be a single directory name")

    seed = raw.get("seed", 3407)
    if not isinstance(seed, int):
        raise ConfigurationError("'seed' must be an integer")

    data_raw = _mapping(raw.get("data", {}), "data")
    train_raw = _mapping(raw.get("train", {}), "train")
    model_raw = _mapping(raw.get("model", {}), "model")
    _reject_unknown(data_raw, {"cache_long_side", "input_size", "num_workers"}, "data")
    _reject_unknown(
        train_raw,
        {
            "max_epochs",
            "micro_batch",
            "grad_accum",
            "mixed_precision",
            "early_stop_patience",
            "adaptation_epochs",
        },
        "train",
    )
    _reject_unknown(model_raw, {"name", "lora_rank", "lora_alpha"}, "model")

    data = DataConfig(**data_raw)
    train = TrainConfig(**train_raw)
    model = ModelConfig(**model_raw)
    if data.cache_long_side <= 0 or data.input_size <= 0 or data.num_workers < 0:
        raise ConfigurationError("data values must be positive (num_workers may be zero)")
    if (
        train.max_epochs < 1
        or train.micro_batch < 1
        or train.grad_accum < 1
        or train.adaptation_epochs < 1
    ):
        raise ConfigurationError(
            "max_epochs, micro_batch, grad_accum and adaptation_epochs must be at least 1"
        )
    if train.early_stop_patience < 1:
        raise ConfigurationError("early_stop_patience must be at least 1")
    if train.mixed_precision is not LOCKED_MIXED_PRECISION:
        raise ConfigurationError("train.mixed_precision is fixed to true for the locked AMP training plan")
    if (
        model.name != LOCKED_MODEL_NAME
        or model.lora_rank != LOCKED_LORA_RANK
        or model.lora_alpha != LOCKED_LORA_ALPHA
    ):
        raise ConfigurationError(
            "model.name, model.lora_rank, and model.lora_alpha are fixed by the locked DINOv3 LoRA plan"
        )
    return AppConfig(dataset=dataset, seed=seed, data=data, train=train, model=model)


def load_config(path: Path) -> AppConfig:
    """Read one task YAML. The filename stem, not YAML, supplies the task name."""
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Configuration must be a .yaml or .yml file")
    if not path.is_file():
        raise ConfigurationError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigurationError("Configuration is empty")
    return parse_config(_mapping(raw, "root"))


