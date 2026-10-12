"""Configuration parsing and strict validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
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
    num_workers: int = 4
    val_fraction: float = 0.2
    calibration_fraction: float = 0.0
    test_fraction: float = 0.0
    split_growth: str = "stratified"
    crop_min_scale: float = 0.70
    color_jitter_strength: float = 1.0
    horizontal_flip_probability: float = 0.5
    grayscale_probability: float = 0.05
    erasing_probability: float = 0.15
    preserve_aspect_ratio: bool = False
    eval_batch_size: int = 32
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    crop_max_scale: float = 1.0
    crop_min_ratio: float = 0.90
    crop_max_ratio: float = 1.10
    color_jitter_probability: float = 0.8
    color_jitter_brightness: float = 0.20
    color_jitter_contrast: float = 0.20
    color_jitter_saturation: float = 0.20
    color_jitter_hue: float = 0.08
    erasing_min_scale: float = 0.02
    erasing_max_scale: float = 0.12


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 50
    micro_batch: int = 8
    grad_accum: int = 4
    mixed_precision: bool = LOCKED_MIXED_PRECISION
    early_stop_patience: int = 10
    adaptation_epochs: int = 12
    class_sampling: str = "sqrt"
    effective_number_beta: float = 0.999
    lora_learning_rate: float = 1e-4
    head_learning_rate: float = 3e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_epochs: int = 3
    minimum_learning_rate: float = 1e-6
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    log_interval: int = 10
    arcface_loss_weight: float = 0.75
    supervised_contrastive_loss_weight: float = 0.25
    supervised_contrastive_temperature: float = 0.07
    lora_unfreeze_epoch: int = 4
    adaptation_learning_rate_multiplier: float = 0.25
    early_stop_min_delta: float = 0.002
    no_validation_min_delta: float = 0.001
    no_validation_early_stop_patience: int = 8
    early_stop_min_epochs: int = 10



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


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inference settings that are independent of the selected dataset.

    ``batch_size`` is intentionally large by default: the ViT-B/16 backbone
    fits comfortably into modern GPU memory at 256×256 resolution and a larger
    batch is the cheapest way to keep the GPU busy during the forward pass.

    ``decode_workers`` uses ``0`` as an explicit auto-detect sentinel meaning
    "use every available CPU core" (capped by the image count at runtime).
    Callers that prefer an explicit thread count may still provide it; the
    previous default of a fixed small value could not exploit the many-core
    hosts where decoding dominates wall time.
    """

    outoftag_probability_threshold: float = 0.20
    outoftag_copy_limit: int = 300
    batch_size: int = 256
    copy_workers: int = 4
    decode_workers: int = 0
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    mixed_precision: bool = True


def load_inference_config(path: Path) -> InferenceConfig:
    """Load and validate shared inference settings from YAML.

    :param path: Path to the inference-only YAML file.
    :return: Immutable inference settings.
    :raises ConfigurationError: If the file is missing or contains invalid settings.
    """
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Inference configuration must be a .yaml or .yml file")
    if not path.is_file():
        raise ConfigurationError(f"Inference configuration not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Inference configuration is not valid YAML: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Inference configuration must contain a YAML mapping")
    _reject_unknown(raw, set(InferenceConfig.__dataclass_fields__), "inference")
    config = InferenceConfig(**raw)
    threshold = config.outoftag_probability_threshold
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ConfigurationError("outoftag_probability_threshold must be in [0, 1]")
    if type(config.outoftag_copy_limit) is not int or config.outoftag_copy_limit < 0:
        raise ConfigurationError("outoftag_copy_limit must be a non-negative integer")
    for name in ("batch_size", "copy_workers"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"{name} must be a positive integer")
    # ``decode_workers == 0`` is the documented auto-detect sentinel.
    if type(config.decode_workers) is not int or config.decode_workers < 0:
        raise ConfigurationError(
            "decode_workers must be a non-negative integer (0 = auto-detect from CPU count)"
        )
    for name in ("allow_tf32", "cudnn_benchmark", "mixed_precision"):
        if type(getattr(config, name)) is not bool:
            raise ConfigurationError(f"{name} must be a boolean")
    return config


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
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ConfigurationError("'seed' must be an integer in [0, 2**32)")

    data_raw = _mapping(raw.get("data", {}), "data")
    train_raw = _mapping(raw.get("train", {}), "train")
    model_raw = _mapping(raw.get("model", {}), "model")
    _reject_unknown(data_raw, set(DataConfig.__dataclass_fields__), "data")
    _reject_unknown(train_raw, set(TrainConfig.__dataclass_fields__), "train")
    _reject_unknown(model_raw, {"name", "lora_rank", "lora_alpha"}, "model")

    data = DataConfig(**data_raw)
    train = TrainConfig(**train_raw)
    model = ModelConfig(**model_raw)
    for section, names in (
        (data, ("cache_long_side", "input_size", "num_workers", "eval_batch_size", "prefetch_factor")),
        (train, (
            "max_epochs", "micro_batch", "grad_accum", "early_stop_patience", "adaptation_epochs",
            "warmup_epochs", "log_interval", "lora_unfreeze_epoch",
            "no_validation_early_stop_patience", "early_stop_min_epochs",
        )),
    ):
        for name in names:
            if type(getattr(section, name)) is not int:
                raise ConfigurationError(f"{name} must be an integer")
    for name in (
        "val_fraction", "calibration_fraction", "test_fraction", "crop_min_scale",
        "color_jitter_strength", "horizontal_flip_probability", "grayscale_probability",
        "erasing_probability",
        "crop_max_scale", "color_jitter_probability", "color_jitter_hue",
        "erasing_min_scale", "erasing_max_scale",
    ):
        value = getattr(data, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ConfigurationError(f"data.{name} must be a finite number in [0, 1]")
    if data.crop_min_scale == 0:
        raise ConfigurationError("data.crop_min_scale must be greater than zero")
    if data.val_fraction + data.calibration_fraction + data.test_fraction >= 1:
        raise ConfigurationError("Holdout fractions must sum to less than 1 to retain training data")
    if data.split_growth not in ("stratified", "train_only"):
        raise ConfigurationError("data.split_growth must be 'stratified' or 'train_only'")
    if type(data.preserve_aspect_ratio) is not bool:
        raise ConfigurationError("data.preserve_aspect_ratio must be a boolean")
    for name in ("pin_memory", "persistent_workers"):
        if type(getattr(data, name)) is not bool:
            raise ConfigurationError(f"data.{name} must be a boolean")
    if not data.crop_min_scale <= data.crop_max_scale <= 1:
        raise ConfigurationError("data crop scale must satisfy 0 < crop_min_scale <= crop_max_scale <= 1")
    if not 0 < data.crop_min_ratio <= data.crop_max_ratio:
        raise ConfigurationError("data crop ratio must satisfy 0 < crop_min_ratio <= crop_max_ratio")
    for name in ("color_jitter_brightness", "color_jitter_contrast", "color_jitter_saturation"):
        value = getattr(data, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"data.{name} must be a finite non-negative number")
    if data.color_jitter_hue > 0.5:
        raise ConfigurationError("data.color_jitter_hue must be at most 0.5")
    if not 0 < data.erasing_min_scale <= data.erasing_max_scale <= 1:
        raise ConfigurationError("data erasing scale must satisfy 0 < min <= max <= 1")
    if train.class_sampling not in ("sqrt", "uniform", "effective_number"):
        raise ConfigurationError("train.class_sampling must be sqrt, uniform, or effective_number")
    beta = train.effective_number_beta
    if type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta < 1:
        raise ConfigurationError("train.effective_number_beta must be in [0, 1)")
    if (data.cache_long_side <= 0 or data.input_size <= 0 or data.num_workers < 0
            or data.eval_batch_size <= 0 or data.prefetch_factor <= 0):
        raise ConfigurationError("data values must be positive (num_workers may be zero)")
    if (
        train.max_epochs < 1
        or train.micro_batch < 1
        or train.grad_accum < 1
        or train.adaptation_epochs < 1
        or train.log_interval < 1
        or train.warmup_epochs < 0
        or train.lora_unfreeze_epoch < 1
    ):
        raise ConfigurationError(
            "max_epochs, micro_batch, grad_accum and adaptation_epochs must be at least 1"
        )
    if train.early_stop_patience < 1:
        raise ConfigurationError("early_stop_patience must be at least 1")
    for name in (
        "lora_learning_rate", "head_learning_rate", "minimum_learning_rate", "gradient_clip_norm",
        "supervised_contrastive_temperature", "adam_epsilon",
    ):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"train.{name} must be a finite positive number")
    for name in ("weight_decay",):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"train.{name} must be a finite non-negative number")
    for name in ("adam_beta1", "adam_beta2", "ema_decay"):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ConfigurationError(f"train.{name} must be a finite number in [0, 1)")
    value = train.adaptation_learning_rate_multiplier
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError("train.adaptation_learning_rate_multiplier must be a finite positive number")
    for name in ("arcface_loss_weight", "supervised_contrastive_loss_weight"):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"train.{name} must be a finite non-negative number")
    for name in ("early_stop_min_delta", "no_validation_min_delta"):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"train.{name} must be a finite non-negative number")
    if train.no_validation_early_stop_patience < 1 or train.early_stop_min_epochs < 1:
        raise ConfigurationError("early stopping patience and minimum epochs must be at least 1")
    if not math.isclose(train.arcface_loss_weight + train.supervised_contrastive_loss_weight, 1.0):
        raise ConfigurationError("train loss weights must sum to 1")
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
