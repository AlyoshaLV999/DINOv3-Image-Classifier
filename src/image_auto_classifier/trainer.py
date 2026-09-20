"""End-to-end deterministic training and resumable model registration."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
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
    latest_epoch_checkpoint,
    load_checkpoint,
    register_best_model,
    save_epoch_checkpoint,
)
from .common import (
    ProjectError,
    capture_rng_state,
    manifest_sha256,
    require_cuda,
    restore_rng_state,
    seed_everything,
)
from .config import AppConfig
from .data import (
    CachedImageDataset,
    ManifestItem,
    SqrtClassSampler,
    audit_dataset,
    build_loader,
    build_transforms,
)
from .losses import supervised_contrastive_loss
from .metrics import Evaluation, build_inference_artifacts, collect_evaluation
from .model import MetricArcFaceModel, ParameterEMA


class EpochLRScheduler:
    """Three-epoch linear warm-up followed by a cosine decay to 1e-6."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.max_epochs = max_epochs
        self.last_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not 1 <= epoch <= self.max_epochs:
            raise ProjectError(f"Epoch {epoch} is outside configured range 1..{self.max_epochs}")
        if epoch <= 3:
            factor = epoch / 3.0
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base_lr * factor
        else:
            progress = (epoch - 3) / max(1, self.max_epochs - 3)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = 1e-6 + (base_lr - 1e-6) * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.last_epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"base_lrs": self.base_lrs, "max_epochs": self.max_epochs, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"base_lrs", "max_epochs", "last_epoch"}:
            raise ProjectError("Checkpoint scheduler state is invalid")
        saved_base_lrs = [float(value) for value in state["base_lrs"]]
        if len(saved_base_lrs) != len(self.base_lrs) or any(
            not math.isclose(saved, current) for saved, current in zip(saved_base_lrs, self.base_lrs, strict=True)
        ):
            raise ProjectError("Checkpoint optimizer learning-rate groups differ from the current configuration")
        self.last_epoch = int(state["last_epoch"])


def _parameter_groups(model: MetricArcFaceModel) -> list[dict[str, Any]]:
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
        decay = 0.0 if id(parameter) in no_decay_ids or name.endswith(".bias") else 0.05
        groups[(kind, decay)].append(parameter)
    parameter_groups: list[dict[str, Any]] = []
    for (kind, decay), parameters in groups.items():
        if parameters:
            parameter_groups.append(
                {"params": parameters, "lr": 1e-4 if kind == "lora" else 3e-4, "weight_decay": decay}
            )
    if not parameter_groups:
        raise ProjectError("No trainable parameters were found")
    return parameter_groups


def _build_optimizer(model: MetricArcFaceModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(_parameter_groups(model), betas=(0.9, 0.999))


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


def _plot_curves(path: Path, history: Mapping[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, history["train_loss"], label="train loss")
    axis.plot(epochs, history["val_loss"], label="val loss")
    axis.plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axis.set_xlabel("epoch")
    axis.set_ylabel("value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _resume_compatibility(config: AppConfig, checkpoint: Mapping[str, Any]) -> None:
    previous = checkpoint.get("config")
    if not isinstance(previous, Mapping):
        raise ProjectError("Checkpoint configuration is invalid")
    if previous.get("dataset") != config.dataset or previous.get("model") != config.to_dict()["model"]:
        raise ProjectError("Dataset and model/LoRA configuration must not change when resuming")
    if previous.get("data") != config.to_dict()["data"]:
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
        resume_path = latest_epoch_checkpoint(task_models_dir)
        if resume_path is None:
            raise ProjectError(f"--resume auto found no epoch checkpoint in {task_models_dir}")
        resume_checkpoint = load_checkpoint(resume_path, device)
        _resume_compatibility(config, resume_checkpoint)
        logger.info("Loaded resume checkpoint from %s", resume_path)

    manifest_items, current_tags = audit_dataset(
        project_root=project_root,
        dataset=config.dataset,
        task_log_dir=task_log_dir,
        cache_long_side=config.data.cache_long_side,
        logger=logger,
    )
    manifest_path = task_log_dir / "split_manifest.jsonl"
    split_hash = manifest_sha256(manifest_path)
    adaptation_mode = False
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

    tag_to_id, new_tags = _tag_mapping(current_tags, model_checkpoint)
    train_items = [item for item in manifest_items if item.split == "train"]
    val_items = [item for item in manifest_items if item.split == "val"]
    if not train_items:
        raise ProjectError("The immutable split manifest has no training images")
    train_transform, eval_transform = build_transforms(config.data.input_size)
    train_eval_dataset = CachedImageDataset(
        project_root, train_items, tag_to_id, eval_transform, two_views=False
    )
    val_dataset = (
        CachedImageDataset(project_root, val_items, tag_to_id, eval_transform, two_views=False)
        if val_items
        else None
    )
    eval_batch_size = 32
    train_eval_loader = build_loader(
        train_eval_dataset,
        batch_size=eval_batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
        shuffle=False,
    )
    val_loader = (
        build_loader(
            val_dataset,
            batch_size=eval_batch_size,
            num_workers=config.data.num_workers,
            seed=config.seed + 1,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )

    model = MetricArcFaceModel(config.model, len(tag_to_id)).to(device)
    ema = ParameterEMA(model, decay=0.999)
    start_epoch = 1
    best_score = -math.inf if val_loader is not None else math.inf
    bad_epochs = 0
    train_loss_ema: float | None = None
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    class_accuracy_threshold = 0.95
    active_class_ids = set(range(len(tag_to_id)))
    old_class_count = 0

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
        for key in history:
            values = loaded_history.get(key, [])
            if not isinstance(values, list):
                raise ProjectError(f"Checkpoint history '{key}' is invalid")
            history[key] = [float(value) for value in values]

        if not (adaptation_mode and not resuming_adaptation):
            raw_active_class_ids = prior_monitor.get("active_class_ids")
            if raw_active_class_ids is not None:
                if not isinstance(raw_active_class_ids, list) or any(
                    type(class_id) is not int for class_id in raw_active_class_ids
                ):
                    raise ProjectError("Checkpoint active class ids are invalid")
                valid_class_ids = set(range(len(tag_to_id)))
                if not set(raw_active_class_ids) <= valid_class_ids:
                    raise ProjectError("Checkpoint active class ids are out of range")
                active_class_ids = set(raw_active_class_ids)
                previous_ids = _validate_tag_to_id(resume_checkpoint["tag_to_id"])
                active_class_ids.update(valid_class_ids - set(previous_ids.values()))

    if adaptation_mode and not resuming_adaptation:
        best_score = -math.inf if val_loader is not None else math.inf
        bad_epochs = 0
        train_loss_ema = None

    if val_loader is None:
        active_class_ids = set(range(len(tag_to_id)))

    optimizer = _build_optimizer(model)
    if adaptation_mode:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= 0.25

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

    scheduler = EpochLRScheduler(optimizer, schedule_max_epochs)
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

    if start_epoch > end_epoch:
        logger.info("Training is already complete at epoch %d; nothing to do.", start_epoch - 1)
        return

    for epoch in range(start_epoch, end_epoch + 1):
        schedule_epoch = epoch - cycle_start_epoch + 1 if adaptation_mode else epoch

        active_train_items = [
            item for item in train_items if tag_to_id[item.tag] in active_class_ids
        ]
        if not active_train_items:
            logger.info(
                "All classes have reached at least %.1f%% validation accuracy; training is complete",
                class_accuracy_threshold * 100.0,
            )
            break

        train_dataset = CachedImageDataset(
            project_root, active_train_items, tag_to_id, train_transform, two_views=True
        )
        sampler = SqrtClassSampler(
            active_train_items, tag_to_id, config.seed, config.train.micro_batch
        )

        model.train()
        model.set_lora_trainable(schedule_epoch >= 4)
        scheduler.set_epoch(schedule_epoch)
        sampler.set_epoch(epoch)

        train_loader = build_loader(
            train_dataset,
            batch_size=config.train.micro_batch,
            num_workers=config.data.num_workers,
            seed=config.seed + epoch,
            sampler=sampler,
            shuffle=False,
        )

        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        arc_losses: list[float] = []
        supcon_losses: list[float] = []
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
                        temperature=0.07,
                    )
                    unscaled_loss = 0.75 * arc_loss + 0.25 * supcon_loss
                    loss = unscaled_loss / config.train.grad_accum

                scaler.scale(loss).backward()
                is_last = batch_index == len(train_loader)
                if batch_index % config.train.grad_accum == 0 or is_last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)

                losses.append(float(unscaled_loss.detach().item()))
                arc_losses.append(float(arc_loss.detach().item()))
                supcon_losses.append(float(supcon_loss.detach().item()))
                progress.set_postfix(
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    loss=f"{losses[-1]:.4f}",
                    loss_arc=f"{arc_losses[-1]:.4f}",
                    loss_supcon=f"{supcon_losses[-1]:.4f}",
                    gpu_mem=f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}G",
                )
        except torch.OutOfMemoryError as error:
            logger.exception(
                "CUDA OOM with micro_batch=%d grad_accum=%d",
                config.train.micro_batch,
                config.train.grad_accum,
            )
            raise ProjectError(
                "CUDA out of memory. Set train.micro_batch from "
                f"{config.train.micro_batch} to 6 and resume from the same checkpoint."
            ) from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                logger.exception(
                    "CUDA OOM with micro_batch=%d grad_accum=%d",
                    config.train.micro_batch,
                    config.train.grad_accum,
                )
                raise ProjectError(
                    "CUDA out of memory. Set train.micro_batch from "
                    f"{config.train.micro_batch} to 6 and resume from the same checkpoint."
                ) from error
            raise

        if not losses:
            raise ProjectError("Training epoch produced no batches")

        train_loss = sum(losses) / len(losses)
        train_evaluation = collect_evaluation(
            model,
            ema,
            train_eval_loader,
            device,
            amp_enabled=config.train.mixed_precision,
        )
        val_evaluation: Evaluation | None = (
            collect_evaluation(
                model,
                ema,
                val_loader,
                device,
                amp_enabled=config.train.mixed_precision,
            )
            if val_loader is not None
            else None
        )

        next_active_class_ids = set(range(len(tag_to_id)))
        class_val_accuracy: dict[str, float] = {}

        if val_evaluation is not None:
            predictions = val_evaluation.logits.argmax(dim=1)

            for tag, class_id in sorted(tag_to_id.items(), key=lambda pair: pair[1]):
                class_mask = val_evaluation.labels.eq(class_id)

                if not bool(class_mask.any().item()):
                    class_val_accuracy[tag] = 0.0
                    continue

                accuracy = float(
                    predictions[class_mask]
                    .eq(val_evaluation.labels[class_mask])
                    .float()
                    .mean()
                    .item()
                )
                class_val_accuracy[tag] = accuracy

                if accuracy >= class_accuracy_threshold:
                    next_active_class_ids.discard(class_id)

            frozen_tags = [
                tag
                for tag, class_id in sorted(tag_to_id.items(), key=lambda pair: pair[1])
                if class_id not in next_active_class_ids
            ]
            active_tags = [
                tag
                for tag, class_id in sorted(tag_to_id.items(), key=lambda pair: pair[1])
                if class_id in next_active_class_ids
            ]

            logger.info(
                "epoch=%d class training gate: active=%s frozen=%s val_accuracy=%s",
                epoch,
                active_tags,
                frozen_tags,
                {
                    tag: round(accuracy, 4)
                    for tag, accuracy in class_val_accuracy.items()
                },
            )
        else:
            next_active_class_ids = set(range(len(tag_to_id)))

        all_classes_satisfied = (
            val_evaluation is not None and not next_active_class_ids
        )
        active_class_ids = next_active_class_ids

        history["train_loss"].append(train_loss)
        history["val_loss"].append(
            val_evaluation.loss if val_evaluation is not None else float("nan")
        )
        history["val_macro_f1"].append(
            val_evaluation.macro_f1 if val_evaluation is not None else float("nan")
        )
        _plot_curves(task_log_dir / "curves.png", history)

        inference = build_inference_artifacts(
            train_evaluation,
            val_evaluation,
            tag_to_id,
        )

        if val_evaluation is not None:
            score = val_evaluation.macro_f1
            improved = score > best_score + 0.002
            patience = config.train.early_stop_patience
            can_stop = schedule_epoch >= 10
            monitor_name = "val_macro_f1"
        else:
            train_loss_ema = (
                train_loss
                if train_loss_ema is None
                else (train_loss + 4.0 * train_loss_ema) / 5.0
            )
            score = train_loss_ema
            improved = score < best_score - 0.001
            patience = 8
            can_stop = True
            monitor_name = "train_loss_ema5"

        if adaptation_mode:
            patience = min(patience, max(3, schedule_max_epochs // 3))
            can_stop = schedule_epoch >= min(6, schedule_max_epochs)

        if improved:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1

        metric_early_stop = can_stop and bad_epochs >= patience
        should_stop = all_classes_satisfied or (
            val_evaluation is None and metric_early_stop
        )
        publish_model = improved or all_classes_satisfied

        monitor: dict[str, Any] = {
            "name": monitor_name,
            "bad_epochs": bad_epochs,
            "train_loss_ema": train_loss_ema,
            "active_class_ids": sorted(active_class_ids),
            "class_val_accuracy": class_val_accuracy,
            "class_accuracy_threshold": class_accuracy_threshold,
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
            inference=inference,
            history=history,
            monitor=monitor,
        )
        epoch_path = save_epoch_checkpoint(
            task_models_dir,
            epoch,
            checkpoint_payload,
        )

        f1_display = (
            val_evaluation.macro_f1
            if val_evaluation is not None
            else float("nan")
        )
        logger.info(
            "epoch=%d lr=%.3e loss_arc=%.5f loss_supcon=%.5f loss=%.5f f1=%s gpu_mem=%.2fGiB",
            epoch,
            optimizer.param_groups[0]["lr"],
            sum(arc_losses) / len(arc_losses),
            sum(supcon_losses) / len(supcon_losses),
            train_loss,
            f"{f1_display:.5f}" if val_evaluation is not None else "n/a",
            torch.cuda.max_memory_allocated(device) / 2**30,
        )

        if publish_model:
            register_best_model(
                epoch_checkpoint=epoch_path,
                project_root=project_root,
                dataset=config.dataset,
                task_name=task_name,
                tag_to_id=tag_to_id,
                config=config,
                inference=inference,
                epoch=epoch,
                split_manifest_hash=split_hash,
                task_models_dir=task_models_dir,
            )
            tqdm.write(
                "[best model]" if improved else "[all classes >=95%]"
            )

        if should_stop:
            if all_classes_satisfied:
                logger.info(
                    "Stopping at epoch %d because every class reached at least %.1f%% validation accuracy",
                    epoch,
                    class_accuracy_threshold * 100.0,
                )
            else:
                logger.info(
                    "Early stopping at epoch %d after %d non-improving epochs",
                    epoch,
                    bad_epochs,
                )
            break


