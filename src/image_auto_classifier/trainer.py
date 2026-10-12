"""End-to-end deterministic training and resumable model registration."""

from __future__ import annotations

import logging
import math
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .checkpointing import (
    build_checkpoint,
    latest_resume_checkpoint,
    load_checkpoint,
    register_best_model,
    save_epoch_checkpoint,
)
from .common import (
    ProjectError,
    capture_rng_state,
    atomic_write_json,
    manifest_sha256,
    require_cuda,
    restore_rng_state,
    seed_everything,
)
from .config import AppConfig, parse_config
from .data import (
    CachedImageDataset,
    ClassBalancedSampler,
    ManifestItem,
    audit_dataset,
    build_loader,
    build_transforms,
)
from .losses import supervised_contrastive_loss
from .metrics import Evaluation, build_inference_artifacts, collect_evaluation
from .model import MetricArcFaceModel, ParameterEMA


class EpochLRScheduler:
    """Linear warm-up followed by a cosine decay to the configured floor."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int,
                 warmup_epochs: int = 3, minimum_learning_rate: float = 1e-6) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.minimum_learning_rate = minimum_learning_rate
        self.last_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not 1 <= epoch <= self.max_epochs:
            raise ProjectError(f"Epoch {epoch} is outside configured range 1..{self.max_epochs}")
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base_lr * factor
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                floor = min(base_lr, self.minimum_learning_rate)
                group["lr"] = floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.last_epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"base_lrs": self.base_lrs, "max_epochs": self.max_epochs, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"base_lrs", "max_epochs", "last_epoch"}:
            raise ProjectError("Checkpoint scheduler state is invalid")
        saved_base_lrs = [float(value) for value in state["base_lrs"]]
        # Only the number of parameter groups is a structural invariant. Learning-rate
        # values may legitimately differ after an adaptation cycle has finished, because
        # the scheduler rebuilds base_lrs from the normal optimizer before restoring
        # last_epoch; set_epoch() will recompute the actual per-epoch learning rates.
        if len(saved_base_lrs) != len(self.base_lrs):
            raise ProjectError("Checkpoint optimizer learning-rate groups differ from the current configuration")
        self.last_epoch = int(state["last_epoch"])


def _parameter_groups(model: MetricArcFaceModel, config: AppConfig) -> list[dict[str, Any]]:
    no_decay_ids: set[int] = {id(model.arcface.weight)}
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    groups: dict[tuple[str, float], list[nn.Parameter]] = {
        ("lora", 0.05): [],
        ("lora", 0.0): [],
        ("head", 0.05): [],
        ("head", 0.0): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = "lora" if "lora_" in name else "head"
        decay = 0.0 if id(parameter) in no_decay_ids or name.endswith(".bias") else config.train.weight_decay
        groups[(kind, decay)].append(parameter)
    parameter_groups: list[dict[str, Any]] = []
    for (kind, decay), parameters in groups.items():
        if parameters:
            parameter_groups.append(
                {"params": parameters,
                 "lr": config.train.lora_learning_rate if kind == "lora" else config.train.head_learning_rate,
                 "weight_decay": decay}
            )
    if not parameter_groups:
        raise ProjectError("No trainable parameters were found")
    return parameter_groups


def _build_optimizer(model: MetricArcFaceModel, config: AppConfig) -> torch.optim.AdamW:
    optimizer_options: dict[str, Any] = {
        "params": _parameter_groups(model, config),
        "betas": (config.train.adam_beta1, config.train.adam_beta2),
        "eps": config.train.adam_epsilon,
    }
    if torch.cuda.is_available():
        # Fused AdamW reduces per-step Python/kernel-launch overhead on CUDA.
        # Keep a compatibility fallback for older or vendor-patched PyTorch builds.
        try:
            return torch.optim.AdamW(**optimizer_options, fused=True)
        except (TypeError, RuntimeError) as error:
            logging.getLogger(__name__).warning(
                "Fused AdamW is unavailable; falling back to the standard optimizer: %s", error
            )
    return torch.optim.AdamW(**optimizer_options)


def _oom_hint(micro_batch: int) -> str:
    smaller_batch = max(1, micro_batch // 2)
    if smaller_batch < micro_batch:
        return f"Try train.micro_batch={smaller_batch} and resume from the same checkpoint."
    return "The batch is already 1; reduce input_size or free GPU memory."


def _apply_accumulated_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: ParameterEMA,
    sample_count: int,
    gradient_clip_norm: float = 1.0,
) -> None:
    """Apply a sample-averaged gradient update, keeping EMA aligned with real steps."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
    scale_before_step = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if scaler.get_scale() >= scale_before_step:
        ema.update(model)


def _validate_tag_to_id(tag_to_id: Mapping[str, Any]) -> dict[str, int]:
    parsed = {str(tag): int(class_id) for tag, class_id in tag_to_id.items()}
    if len(parsed) != len(set(parsed.values())) or sorted(parsed.values()) != list(range(len(parsed))):
        raise ProjectError("Checkpoint tag_to_id must contain contiguous unique ids beginning at zero")
    return parsed


def _tag_mapping(current_tags: list[str], checkpoint: Mapping[str, Any] | None) -> tuple[dict[str, int], list[str]]:
    if checkpoint is None:
        return {tag: index for index, tag in enumerate(sorted(current_tags))}, []
    previous = _validate_tag_to_id(checkpoint["tag_to_id"])
    removed = sorted(set(previous) - set(current_tags))
    if removed:
        raise ProjectError(
            "Refusing resume after tags were removed, because their classifier/prototype state would be invalid: "
            f"{removed}"
        )
    new_tags = sorted(set(current_tags) - set(previous))
    mapping = dict(previous)
    for tag in new_tags:
        mapping[tag] = len(mapping)
    return mapping, new_tags


def _load_expanded_model_state(
    model: MetricArcFaceModel, checkpoint: Mapping[str, Any], old_class_count: int
) -> None:
    saved_state = checkpoint["model_state"]
    if not isinstance(saved_state, Mapping):
        raise ProjectError("Checkpoint model state is invalid")
    state = dict(saved_state)
    saved_arcface = state.pop("arcface.weight", None)
    if not isinstance(saved_arcface, torch.Tensor) or saved_arcface.shape != (old_class_count, 256):
        raise ProjectError("Checkpoint ArcFace weight has an unexpected shape")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected_missing = set(incompatible.missing_keys) - {"arcface.weight"}
    if unexpected_missing or incompatible.unexpected_keys:
        raise ProjectError(
            "Checkpoint model topology differs from the loaded DINOv3/LoRA topology; "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    with torch.no_grad():
        model.arcface.weight[:old_class_count].copy_(saved_arcface.to(model.arcface.weight.device))


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer, checkpoint: Mapping[str, Any], model: MetricArcFaceModel, old_classes: int) -> None:
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except Exception as error:
        raise ProjectError(f"Checkpoint optimizer state cannot be restored: {error}") from error
    state = optimizer.state.get(model.arcface.weight, {})
    for key, value in list(state.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == old_classes:
            expanded = torch.zeros_like(model.arcface.weight, dtype=value.dtype, device=value.device)
            expanded[:old_classes].copy_(value)
            state[key] = expanded


@torch.no_grad()
def _initialize_new_arcface_rows(
    model: MetricArcFaceModel,
    ema: ParameterEMA,
    loader: torch.utils.data.DataLoader,
    new_tags: list[str],
    tag_to_id: Mapping[str, int],
    device: torch.device,
    amp_enabled: bool,
) -> None:
    if not new_tags:
        return
    was_training = model.training
    model.eval()
    features_by_id: dict[int, list[torch.Tensor]] = {tag_to_id[tag]: [] for tag in new_tags}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            features = model.forward_features(images)
        for class_id in features_by_id:
            mask = labels == class_id
            if mask.any():
                features_by_id[class_id].append(features[mask].float())
    for tag in new_tags:
        class_id = tag_to_id[tag]
        values = features_by_id[class_id]
        if values:
            row = F.normalize(torch.cat(values).mean(dim=0), dim=0)
        else:
            row = torch.empty(256, device=device).normal_(mean=0.0, std=0.01)
        model.arcface.weight[class_id].copy_(row)
        ema.shadow["arcface.weight"][class_id].copy_(row)
    if was_training:
        model.train()


HISTORY_KEYS = (
    "train_loss", "train_arc_loss", "train_supcon_loss", "train_eval_ce_loss",
    "val_ce_loss", "val_loss", "val_macro_f1",
)


def _restore_history(raw: Mapping[str, Any], completed_epochs: int) -> dict[str, list[float]]:
    """Align legacy histories without inventing unavailable per-epoch measurements."""
    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ProjectError("Checkpoint completed epoch count is invalid")
    history: dict[str, list[float]] = {}
    for key in HISTORY_KEYS:
        values = raw.get(key, raw.get("val_loss", []) if key == "val_ce_loss" else [])
        if not isinstance(values, list) or len(values) > completed_epochs:
            raise ProjectError(f"Checkpoint history '{key}' is invalid")
        try:
            parsed = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise ProjectError(f"Checkpoint history '{key}' contains invalid values") from error
        history[key] = parsed + [float("nan")] * (completed_epochs - len(parsed))
    return history


def _monitor_step(
    score: float, best_score: float, reference_score: float, bad_epochs: int, *,
    maximize: bool, min_delta: float, patience: int, can_stop: bool,
) -> tuple[float, float, int, bool, bool]:
    """Select the absolute best model; apply min_delta only to patience tracking."""
    if not math.isfinite(score):
        raise ProjectError("Training monitor produced a non-finite score")
    improved = score > best_score if maximize else score < best_score
    significant = score > reference_score + min_delta if maximize else score < reference_score - min_delta
    if improved:
        best_score = score
    if significant:
        reference_score, bad_epochs = score, 0
    else:
        bad_epochs += 1
    return best_score, reference_score, bad_epochs, improved, can_stop and bad_epochs >= patience


def _class_counts(items: Sequence[ManifestItem], tag_to_id: Mapping[str, int]) -> dict[str, int]:
    """Return per-tag sample counts keyed in stable class-id order.

    The returned mapping is ordered by ascending class id so that log lines are
    stable across runs and directly comparable between tasks. Tags absent from
    ``items`` are reported as zero, which makes empty classes visible instead
    of silently disappearing from the log line.

    :param items: Any sequence of manifest items (whole dataset or one split).
    :param tag_to_id: Tag-to-class-id mapping used to fix ordering.
    :return: Ordered ``{tag: count}`` mapping covering every registered tag.
    """
    counts = Counter(item.tag for item in items)
    return {tag: counts[tag] for tag in sorted(tag_to_id, key=lambda tag: tag_to_id[tag])}


def _plot_curves(
    path: Path,
    history: Mapping[str, list[float]],
    *,
    best_epochs: Sequence[int] = (),
) -> None:
    """Render the three-panel training dashboard.

    :param path: Destination PNG path.
    :param history: Per-epoch metric history keyed by metric name.
    :param best_epochs: 1-based epoch indices at which a new best model was
        selected. Each is marked on every panel with a red dashed vertical
        line at ``x = epoch`` and annotated with its epoch number at the top
        of the first panel, matching the requirement that every epoch equal
        to the running best is highlighted. Duplicates (which should not
        normally occur) are collapsed via ``set`` to keep the plot stable.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    try:
        for key in ("train_loss", "train_arc_loss", "train_supcon_loss"):
            axes[0].plot(epochs, history[key], label=key)
        axes[0].set_title("Training objective: augmented views, ArcFace margin + SupCon")
        for key in ("train_eval_ce_loss", "val_ce_loss"):
            axes[1].plot(epochs, history[key], label=key)
        axes[1].set_title("Comparable CE: same EMA, eval transform, no angular margin")
        axes[2].plot(epochs, history["val_macro_f1"], label="val_macro_f1")
        axes[2].set_ylim(0, 1)

        for best_epoch in sorted(set(best_epochs)):
            for axis in axes:
                axis.axvline(
                    best_epoch,
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
            # Anchor the label to the top of the first panel so the epoch
            # number stays aligned with its vertical line across every axis.
            axes[0].annotate(
                str(best_epoch),
                xy=(best_epoch, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                va="top",
                color="red",
                fontsize=9,
            )

        for axis, label in zip(axes, ("objective loss", "cross entropy", "macro-F1"), strict=True):
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.legend()
        axes[2].set_xlabel("epoch")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _resume_compatibility(config: AppConfig, checkpoint: Mapping[str, Any]) -> None:
    previous = checkpoint.get("config")
    if not isinstance(previous, Mapping):
        raise ProjectError("Checkpoint configuration is invalid")
    if previous.get("dataset") != config.dataset or previous.get("model") != config.to_dict()["model"]:
        raise ProjectError("Dataset and model/LoRA configuration must not change when resuming")
    if parse_config(previous).data != config.data:
        raise ProjectError("Data preprocessing configuration must not change when resuming")


def train(
    *,
    project_root: Path,
    task_name: str,
    config: AppConfig,
    resume_auto: bool,
    logger: logging.Logger,
) -> None:
    """Run one task with exact resume or bounded adaptation after dataset changes."""
    device = require_cuda()
    seed_everything(config.seed)
    task_log_dir = project_root / "logs" / task_name
    task_models_dir = task_log_dir / "models"
    task_models_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: dict[str, Any] | None = None
    if resume_auto:
        resume_path = latest_resume_checkpoint(task_models_dir)
        if resume_path is None:
            logger.info("No checkpoint found in %s; starting a new training run", task_models_dir)
        else:
            resume_checkpoint = load_checkpoint(resume_path, device)
            _resume_compatibility(config, resume_checkpoint)
            logger.info("Loaded resume checkpoint from %s", resume_path)

    manifest_items, current_tags = audit_dataset(
        project_root=project_root,
        dataset=config.dataset,
        task_log_dir=task_log_dir,
        cache_long_side=config.data.cache_long_side,
        logger=logger,
        split_config=config.data,
    )
    manifest_path = task_log_dir / "split_manifest.jsonl"
    split_hash = manifest_sha256(manifest_path)
    adaptation_mode = False
    already_stopped = False
    resuming_adaptation = False
    adaptation_start_epoch: int | None = None
    adaptation_end_epoch: int | None = None
    model_checkpoint = resume_checkpoint
    if resume_checkpoint is not None:
        _tag_mapping(current_tags, resume_checkpoint)
        previous_split_hash = resume_checkpoint["split_manifest_hash"]
        if not isinstance(previous_split_hash, str):
            raise ProjectError("Checkpoint split manifest hash is invalid")
        prior_monitor = resume_checkpoint["monitor"]
        if not isinstance(prior_monitor, Mapping):
            raise ProjectError("Checkpoint monitor state is invalid")
        saved_adaptation = prior_monitor.get("adaptation")
        if saved_adaptation is not None:
            if not isinstance(saved_adaptation, Mapping):
                raise ProjectError("Checkpoint adaptation state is invalid")
            saved_start = saved_adaptation.get("start_epoch")
            saved_end = saved_adaptation.get("end_epoch")
            completed = saved_adaptation.get("completed")
            if (
                type(saved_start) is not int
                or type(saved_end) is not int
                or not isinstance(completed, bool)
                or saved_start < 1
                or saved_end < saved_start
            ):
                raise ProjectError("Checkpoint adaptation state is invalid")
            if not completed and split_hash == previous_split_hash:
                if int(resume_checkpoint["epoch"]) >= saved_end:
                    raise ProjectError("Checkpoint adaptation state is inconsistent with its epoch")
                adaptation_mode = True
                resuming_adaptation = True
                adaptation_start_epoch = saved_start
                adaptation_end_epoch = saved_end
        if split_hash != previous_split_hash:
            adaptation_mode = True
            resuming_adaptation = False
            best_path = task_models_dir / "best.ckpt"
            if best_path.is_file():
                model_checkpoint = load_checkpoint(best_path, device)
                _resume_compatibility(config, model_checkpoint)
                logger.info(
                    "Dataset change detected; starting a %d-epoch adaptation cycle from the prior best model %s",
                    config.train.adaptation_epochs,
                    best_path,
                )
            else:
                logger.warning(
                    "Dataset change detected but %s is unavailable; adapting from the latest checkpoint instead",
                    best_path,
                )
        elif resuming_adaptation:
            logger.info(
                "Resuming adaptation cycle ending at epoch %d from the latest checkpoint",
                adaptation_end_epoch,
            )
        elif (
            prior_monitor.get("stopped_early") is True
            and parse_config(resume_checkpoint["config"]).train == config.train
            and resume_checkpoint["config"].get("seed", 3407) == config.seed
        ):
            logger.info("This unchanged run already stopped early at epoch %d", resume_checkpoint["epoch"])
            already_stopped = True

    tag_to_id, new_tags = _tag_mapping(current_tags, model_checkpoint)
    train_items = [item for item in manifest_items if item.split == "train"]
    val_items = [item for item in manifest_items if item.split == "val"]
    calibration_items = [item for item in manifest_items if item.split == "calibration"]
    test_items = [item for item in manifest_items if item.split == "test"]
    if not train_items:
        raise ProjectError("The immutable split manifest has no training images")

    # Per-class sample counts are logged once, before any epoch runs, so that
    # class imbalance, empty classes, and split skew are immediately visible
    # in the run log. Counts are keyed in class-id order for stable output.
    logger.info(
        "class counts at training start | total=%s train=%s val=%s calibration=%s test=%s",
        _class_counts(manifest_items, tag_to_id),
        _class_counts(train_items, tag_to_id),
        _class_counts(val_items, tag_to_id),
        _class_counts(calibration_items, tag_to_id),
        _class_counts(test_items, tag_to_id),
    )

    train_transform, eval_transform = build_transforms(config.data.input_size, config=config.data)
    train_eval_dataset = CachedImageDataset(
        project_root, train_items, tag_to_id, eval_transform, two_views=False
    )
    val_dataset = (
        CachedImageDataset(project_root, val_items, tag_to_id, eval_transform, two_views=False)
        if val_items
        else None
    )
    eval_batch_size = config.data.eval_batch_size
    train_eval_loader = build_loader(
        train_eval_dataset,
        batch_size=eval_batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
        shuffle=False,
        persistent_workers=config.data.persistent_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
    )
    val_loader = (
        build_loader(
            val_dataset,
            batch_size=eval_batch_size,
            num_workers=config.data.num_workers,
            seed=config.seed + 1,
            shuffle=False,
            persistent_workers=config.data.persistent_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor,
        )
        if val_dataset is not None
        else None
    )

    calibration_loader = (
        build_loader(
            CachedImageDataset(project_root, calibration_items, tag_to_id, eval_transform, two_views=False),
            batch_size=eval_batch_size, num_workers=config.data.num_workers,
            seed=config.seed + 2, shuffle=False,
            persistent_workers=config.data.persistent_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor,
        ) if calibration_items else None
    )
    if calibration_loader is None:
        logger.warning(
            "No independent calibration split: %s. Validation scores are not independent test results.",
            "using fixed calibration defaults" if config.data.calibration_fraction > 0 or not val_items
            else "validation is also used for temperature/threshold selection",
        )
    if not test_items:
        logger.warning("No independent test split is available; report validation performance only")
    val_counts = Counter(item.tag for item in val_items)
    sparse_tags = {tag: val_counts[tag] for tag in tag_to_id if val_counts[tag] < 5}
    if sparse_tags:
        logger.warning("Validation support below 5 images for these classes: %s", sparse_tags)

    model = MetricArcFaceModel(config.model, len(tag_to_id)).to(device)
    ema = ParameterEMA(model, decay=config.train.ema_decay)
    start_epoch = 1
    best_score = -math.inf if val_loader is not None else math.inf
    bad_epochs = 0
    train_loss_ema: float | None = None
    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    reference_score = best_score
    old_class_count = 0
    # 1-based epochs where a new best model was selected during this run.
    # Populated inside the training loop after ``_monitor_step`` returns
    # ``improved=True`` and consumed by ``_plot_curves`` to draw ``x = epoch``.
    best_epochs: list[int] = []
    best_inference: dict[str, Any] = (
        dict(model_checkpoint["inference"])
        if model_checkpoint is not None and isinstance(model_checkpoint.get("inference"), Mapping)
        else {}
    )

    if model_checkpoint is not None:
        previous_mapping = _validate_tag_to_id(model_checkpoint["tag_to_id"])
        old_class_count = len(previous_mapping)
        _load_expanded_model_state(model, model_checkpoint, old_class_count)
        ema.load_state_dict(model_checkpoint["ema_state"], model, old_class_count)

    if resume_checkpoint is not None:
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_score = float(resume_checkpoint["best_score"])
        prior_monitor = resume_checkpoint["monitor"]
        if not isinstance(prior_monitor, Mapping):
            raise ProjectError("Checkpoint monitor state is invalid")
        bad_epochs = int(prior_monitor.get("bad_epochs", 0))
        raw_ema = prior_monitor.get("train_loss_ema")
        train_loss_ema = float(raw_ema) if raw_ema is not None else None

        loaded_history = resume_checkpoint["history"]
        if not isinstance(loaded_history, Mapping):
            raise ProjectError("Checkpoint history is invalid")
        history = _restore_history(loaded_history, start_epoch - 1)
        reference_score = float(prior_monitor.get("reference_score", best_score))
        if "active_class_ids" in prior_monitor:
            logger.info("Legacy class gate is disabled; all classes participate in training")
        previous_train = parse_config(resume_checkpoint["config"]).train
        if (previous_train.class_sampling, previous_train.effective_number_beta) != (
            config.train.class_sampling, config.train.effective_number_beta
        ):
            bad_epochs = 0
            reference_score = best_score
            logger.info("Sampling strategy changed; resetting early-stop patience")

    if adaptation_mode and not resuming_adaptation:
        best_score = -math.inf if val_loader is not None else math.inf
        bad_epochs = 0
        train_loss_ema = None
        reference_score = best_score

    optimizer = _build_optimizer(model, config)
    if adaptation_mode:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= config.train.adaptation_learning_rate_multiplier

    if adaptation_mode:
        if resuming_adaptation:
            if adaptation_start_epoch is None or adaptation_end_epoch is None:
                raise ProjectError("Checkpoint adaptation state is incomplete")
            cycle_start_epoch = adaptation_start_epoch
            end_epoch = adaptation_end_epoch
        else:
            cycle_start_epoch = start_epoch
            end_epoch = start_epoch + config.train.adaptation_epochs - 1
        schedule_max_epochs = end_epoch - cycle_start_epoch + 1
    else:
        cycle_start_epoch = start_epoch
        end_epoch = config.train.max_epochs
        schedule_max_epochs = config.train.max_epochs

    scheduler = EpochLRScheduler(
        optimizer, schedule_max_epochs,
        warmup_epochs=min(config.train.warmup_epochs, schedule_max_epochs),
        minimum_learning_rate=config.train.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.mixed_precision)

    if model_checkpoint is not None:
        _initialize_new_arcface_rows(
            model,
            ema,
            train_eval_loader,
            new_tags,
            tag_to_id,
            device,
            config.train.mixed_precision,
        )

    if resume_checkpoint is not None:
        if adaptation_mode and not resuming_adaptation:
            restore_rng_state(resume_checkpoint["rng_state"])
        else:
            _restore_optimizer_state(optimizer, resume_checkpoint, model, old_class_count)
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            try:
                scaler.load_state_dict(resume_checkpoint["scaler_state"])
            except Exception as error:
                raise ProjectError(f"Checkpoint GradScaler state cannot be restored: {error}") from error
            restore_rng_state(resume_checkpoint["rng_state"])

    if start_epoch > end_epoch or already_stopped:
        logger.info("Training updates are already complete at epoch %d", start_epoch - 1)
        end_epoch = start_epoch - 1

    for epoch in range(start_epoch, end_epoch + 1):
        schedule_epoch = epoch - cycle_start_epoch + 1 if adaptation_mode else epoch

        train_dataset = CachedImageDataset(
            project_root, train_items, tag_to_id, train_transform, two_views=True
        )
        sampler = ClassBalancedSampler(
            train_items, tag_to_id, config.seed, config.train.micro_batch,
            strategy=config.train.class_sampling, beta=config.train.effective_number_beta,
        )

        model.train()
        model.set_lora_trainable(schedule_epoch >= config.train.lora_unfreeze_epoch)
        scheduler.set_epoch(schedule_epoch)
        sampler.set_epoch(epoch)

        train_loader = build_loader(
            train_dataset,
            batch_size=config.train.micro_batch,
            num_workers=config.data.num_workers,
            seed=config.seed + epoch,
            sampler=sampler,
            shuffle=False,
            persistent_workers=False,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
        batch_count = 0
        sample_count = 0
        accumulation_samples = 0
        log_interval = config.train.log_interval
        torch.cuda.reset_peak_memory_stats(device)
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", unit="batch")

        try:
            for batch_index, (first, second, labels) in enumerate(progress, start=1):
                first = first.to(device, non_blocking=True)
                second = second.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                images = torch.cat((first, second), dim=0)
                view_labels = torch.cat((labels, labels), dim=0)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=config.train.mixed_precision,
                ):
                    features, logits = model(images, view_labels)
                    arc_loss = (
                        F.cross_entropy(logits, view_labels)
                        if len(tag_to_id) > 1
                        else features.new_zeros(())
                    )
                    supcon_loss = supervised_contrastive_loss(
                        features,
                        view_labels,
                        temperature=config.train.supervised_contrastive_temperature,
                    )
                    unscaled_loss = (
                        config.train.arcface_loss_weight * arc_loss
                        + config.train.supervised_contrastive_loss_weight * supcon_loss
                    )
                    # Weight by the actual number of images in the window. The
                    # last accumulation window and final micro-batch may be short.
                    loss = unscaled_loss * labels.numel()

                scaler.scale(loss).backward()
                accumulation_samples += labels.numel()
                is_last = batch_index == len(train_loader)
                if batch_index % config.train.grad_accum == 0 or is_last:
                    _apply_accumulated_step(
                        model, optimizer, scaler, ema, accumulation_samples,
                        config.train.gradient_clip_norm,
                    )
                    accumulation_samples = 0

                loss_sums.add_(
                    torch.stack(
                        (unscaled_loss.detach(), arc_loss.detach(), supcon_loss.detach())
                    ).to(dtype=torch.float64) * labels.numel()
                )
                sample_count += labels.numel()
                batch_count = batch_index
                if batch_index % log_interval == 0 or batch_index == len(train_loader):
                    running_means = (loss_sums / sample_count).cpu().tolist()
                    progress.set_postfix(
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        loss=f"{running_means[0]:.4f}",
                        loss_arc=f"{running_means[1]:.4f}",
                        loss_supcon=f"{running_means[2]:.4f}",
                        gpu_mem=f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}G",
                    )
        except torch.OutOfMemoryError as error:
            logger.exception(
                "CUDA OOM with micro_batch=%d grad_accum=%d",
                config.train.micro_batch,
                config.train.grad_accum,
            )
            raise ProjectError(f"CUDA out of memory. {_oom_hint(config.train.micro_batch)}") from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                logger.exception(
                    "CUDA OOM with micro_batch=%d grad_accum=%d",
                    config.train.micro_batch,
                    config.train.grad_accum,
                )
                raise ProjectError(f"CUDA out of memory. {_oom_hint(config.train.micro_batch)}") from error
            raise

        if batch_count == 0:
            raise ProjectError("Training epoch produced no batches")

        loss_means = (loss_sums / sample_count).cpu().tolist()
        train_loss, arc_loss_mean, supcon_loss_mean = loss_means
        val_evaluation: Evaluation | None = (
            collect_evaluation(
                model,
                ema,
                val_loader,
                device,
                amp_enabled=config.train.mixed_precision,
                description=f"eval[val] e{epoch:03d}",
            )
            if val_loader is not None
            else None
        )

        # Per-class diagnostics are only meaningful for classes with support in
        # this split. Classes that are absent from validation are skipped
        # entirely so their precision/recall/F1 are not misreported as zero,
        # which would otherwise dominate the log output and mislead operators.
        class_val_metrics: dict[str, dict[str, float | int | None]] = {}
        if val_evaluation is not None:
            predictions = val_evaluation.logits.argmax(dim=1)
            for tag, class_id in sorted(tag_to_id.items(), key=lambda pair: pair[1]):
                actual = val_evaluation.labels.eq(class_id)
                support = int(actual.sum().item())
                if support == 0:
                    continue
                predicted = predictions.eq(class_id)
                predicted_count = int(predicted.sum().item())
                correct = int((actual & predicted).sum().item())
                precision = correct / predicted_count if predicted_count else 0.0
                recall = correct / support
                f1 = 2 * correct / (support + predicted_count)
                class_val_metrics[tag] = {
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            missing_classes = len(tag_to_id) - val_evaluation.evaluated_class_count
            if missing_classes:
                # The split does not cover every registered class. Macro/balanced
                # metrics are deliberately computed over present classes only, so
                # callers must not compare them with a hypothetical full-class
                # score; the warning documents that gap.
                logger.warning(
                    "Validation split covers %d/%d classes; %d classes absent and excluded from aggregate metrics",
                    val_evaluation.evaluated_class_count,
                    len(tag_to_id),
                    missing_classes,
                )
            logger.info("epoch=%d per-class validation metrics=%s", epoch, class_val_metrics)

        history["train_arc_loss"].append(arc_loss_mean)
        history["train_supcon_loss"].append(supcon_loss_mean)
        history["val_ce_loss"].append(val_evaluation.loss if val_evaluation is not None else float("nan"))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(
            val_evaluation.loss if val_evaluation is not None else float("nan")
        )
        history["val_macro_f1"].append(
            val_evaluation.macro_f1 if val_evaluation is not None else float("nan")
        )

        if val_evaluation is not None:
            score = val_evaluation.macro_f1
            min_delta = config.train.early_stop_min_delta
            patience = config.train.early_stop_patience
            can_stop = schedule_epoch >= config.train.early_stop_min_epochs
            monitor_name = "val_macro_f1"
        else:
            train_loss_ema = (
                train_loss
                if train_loss_ema is None
                else (train_loss + 4.0 * train_loss_ema) / 5.0
            )
            score = train_loss_ema
            min_delta = config.train.no_validation_min_delta
            patience = config.train.no_validation_early_stop_patience
            can_stop = True
            monitor_name = "train_loss_ema5"

        if adaptation_mode:
            patience = min(patience, max(3, schedule_max_epochs // 3))
            can_stop = schedule_epoch >= min(6, schedule_max_epochs)

        best_score, reference_score, bad_epochs, improved, should_stop = _monitor_step(
            score, best_score, reference_score, bad_epochs,
            maximize=val_evaluation is not None, min_delta=min_delta,
            patience=patience, can_stop=can_stop,
        )
        if improved:
            # Record this epoch so ``_plot_curves`` can render ``x = epoch``.
            # ``improved`` already guarantees the value is strictly better than
            # every earlier observation, so appending is sufficient; duplicates
            # cannot occur within a single run.
            best_epochs.append(epoch)

            # Training features are only used to build the registered best
            # model's prototypes. Re-evaluating the
            # full training split on non-best epochs adds substantial forward
            # pass cost without affecting monitoring or early stopping.
            train_evaluation = collect_evaluation(
                model,
                ema,
                train_eval_loader,
                device,
                amp_enabled=config.train.mixed_precision,
                description=f"eval[train] e{epoch:03d}",
            )
            calibration_evaluation = (
                collect_evaluation(
                    model,
                    ema,
                    calibration_loader,
                    device,
                    amp_enabled=config.train.mixed_precision,
                    description=f"eval[calib] e{epoch:03d}",
                )
                if calibration_loader is not None
                else None
            )
            # Explicit independent-calibration requests never silently fall back
            # to validation.
            calibration_source = (
                "calibration" if calibration_evaluation is not None
                else "defaults" if config.data.calibration_fraction > 0 or val_evaluation is None
                else "validation"
            )
            best_inference = build_inference_artifacts(
                train_evaluation,
                calibration_evaluation if calibration_source == "calibration" else
                val_evaluation if calibration_source == "validation" else None,
                tag_to_id,
            )
            best_inference["calibration_source"] = calibration_source
            train_eval_ce = train_evaluation.loss
        else:
            train_eval_ce = float("nan")
        history["train_eval_ce_loss"].append(train_eval_ce)

        # The plot is regenerated every epoch after monitoring so the current
        # best-epoch set (including today's improvement) is always reflected.
        _plot_curves(task_log_dir / "curves.png", history, best_epochs=best_epochs)

        monitor: dict[str, Any] = {
            "name": monitor_name,
            "bad_epochs": bad_epochs,
            "reference_score": reference_score,
            "train_loss_ema": train_loss_ema,
            "class_val_metrics": class_val_metrics,
            "stopped_early": should_stop,
        }

        if adaptation_mode:
            monitor["adaptation"] = {
                "start_epoch": cycle_start_epoch,
                "end_epoch": end_epoch,
                "completed": epoch >= end_epoch or should_stop,
            }

        checkpoint_payload = build_checkpoint(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler_state=scheduler.state_dict(),
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            rng_state=capture_rng_state(),
            tag_to_id=tag_to_id,
            split_manifest_hash=split_hash,
            config=config,
            inference=best_inference,
            history=history,
            monitor=monitor,
        )
        save_start = time.perf_counter()
        epoch_path = save_epoch_checkpoint(
            task_models_dir,
            epoch,
            checkpoint_payload,
        )
        save_seconds = time.perf_counter() - save_start

        # Report both macro and weighted views so operators can distinguish
        # "every class counts equally" from "every sample counts equally".
        # ``classes_evaluated`` records the actual coverage of the validation
        # split, since macro/weighted/balanced numbers are computed over that
        # subset only.
        train_eval_ce_display = f"{train_eval_ce:.5f}" if math.isfinite(train_eval_ce) else "n/a"
        if val_evaluation is not None:
            val_ce_display = f"{val_evaluation.loss:.5f}"
            val_macro_display = f"{val_evaluation.macro_f1:.5f}"
            val_weighted_display = f"{val_evaluation.weighted_f1:.5f}"
            val_bal_display = f"{val_evaluation.balanced_accuracy:.5f}"
            classes_evaluated = val_evaluation.evaluated_class_count
        else:
            val_ce_display = "n/a"
            val_macro_display = "n/a"
            val_weighted_display = "n/a"
            val_bal_display = "n/a"
            classes_evaluated = 0
        logger.info(
            "epoch=%d lr=%.3e loss_arc=%.5f loss_supcon=%.5f train_objective=%.5f train_eval_ce=%s "
            "val_ce=%s val_macro_f1=%s val_weighted_f1=%s val_bal_acc=%s classes_evaluated=%d/%d "
            "gpu_mem=%.2fGiB ckpt_save_s=%.2f",
            epoch,
            optimizer.param_groups[0]["lr"],
            arc_loss_mean,
            supcon_loss_mean,
            train_loss,
            train_eval_ce_display,
            val_ce_display,
            val_macro_display,
            val_weighted_display,
            val_bal_display,
            classes_evaluated,
            len(tag_to_id),
            torch.cuda.max_memory_allocated(device) / 2**30,
            save_seconds,
        )

        if improved:
            register_start = time.perf_counter()
            register_best_model(
                epoch_checkpoint=epoch_path,
                project_root=project_root,
                dataset=config.dataset,
                task_name=task_name,
                tag_to_id=tag_to_id,
                config=config,
                inference=best_inference,
                epoch=epoch,
                split_manifest_hash=split_hash,
                task_models_dir=task_models_dir,
            )
            tqdm.write(f"[best model] registered in {time.perf_counter() - register_start:.2f}s")

        if should_stop:
            logger.info("Early stopping at epoch %d after %d non-improving epochs", epoch, bad_epochs)
            break

    if test_items:
        # Test is evaluated once, after selection, using the saved best EMA model.
        # It never enters stopping, calibration, or prototypes.
        best = load_checkpoint(task_models_dir / "best.ckpt", device)
        report_path = task_log_dir / "test_metrics.json"
        if start_epoch > end_epoch and report_path.is_file():
            try:
                previous_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Rebuilding unreadable test report %s: %s", report_path, error)
            else:
                if (
                    isinstance(previous_report, dict)
                    and previous_report.get("selected_epoch") == best["epoch"]
                    and previous_report.get("split_manifest_hash") == split_hash
                ):
                    logger.info("The selected model already has a final test report")
                    return
        model.load_state_dict(best["model_state"], strict=True)
        ema.load_state_dict(best["ema_state"], model)
        test_loader = build_loader(
            CachedImageDataset(project_root, test_items, tag_to_id, eval_transform, two_views=False),
            batch_size=eval_batch_size, num_workers=config.data.num_workers,
            seed=config.seed + 3, shuffle=False,
            persistent_workers=config.data.persistent_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor,
        )
        test = collect_evaluation(
            model,
            ema,
            test_loader,
            device,
            amp_enabled=config.train.mixed_precision,
            description="eval[test]",
        )
        report = {
            "selected_epoch": best["epoch"],
            "split_manifest_hash": split_hash,
            "sample_count": len(test_items),
            "ce_loss": test.loss,
            "top1": test.top1,
            "macro_f1": test.macro_f1,
            "weighted_f1": test.weighted_f1,
            "balanced_accuracy": test.balanced_accuracy,
            "precision": test.precision,
            "recall": test.recall,
            "evaluated_class_count": test.evaluated_class_count,
            "class_count": len(tag_to_id),
            "note": "One evaluation of best EMA; do not tune on this test report.",
        }
        atomic_write_json(report_path, report)
        logger.info("Final independent test evaluation: %s", report)
