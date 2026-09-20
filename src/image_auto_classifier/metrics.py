"""EMA evaluation, calibration, class prototypes, and active-learning selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import ProjectError
from .model import MetricArcFaceModel, ParameterEMA


@dataclass(frozen=True)
class Evaluation:
    loss: float
    top1: float
    precision: float
    recall: float
    macro_f1: float
    balanced_accuracy: float
    features: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


@torch.no_grad()
def collect_evaluation(
    model: MetricArcFaceModel,
    ema: ParameterEMA,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> Evaluation:
    """Collect deterministic, no-augmentation EMA features and validation metrics."""
    was_training = model.training
    model.eval()
    features_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    with ema.average_parameters(model), torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                features, logits = model(images, labels=None)
            features_parts.append(features.float().cpu())
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.cpu())
    if was_training:
        model.train()
    if not labels_parts:
        raise ProjectError("Evaluation loader produced no batches")
    features = torch.cat(features_parts)
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    loss = F.cross_entropy(logits, labels).item() if logits.shape[1] > 1 else 0.0
    predictions = logits.argmax(dim=1)
    targets_np = labels.numpy()
    predictions_np = predictions.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions_np, average="macro", zero_division=0
    )

    if np.isin(predictions_np, targets_np).all():
        balanced_accuracy = float(balanced_accuracy_score(targets_np, predictions_np))
    else:
        class_recalls = [
            np.mean(predictions_np[targets_np == class_id] == class_id)
            for class_id in np.unique(targets_np)
        ]
        balanced_accuracy = float(np.mean(class_recalls))

    return Evaluation(
        loss=float(loss),
        top1=float((predictions == labels).float().mean().item()),
        precision=float(precision),
        recall=float(recall),
        macro_f1=float(f1),
        balanced_accuracy=balanced_accuracy,
        features=features,
        logits=logits,
        labels=labels,
    )



def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() < 20 or logits.shape[1] <= 1:
        return 1.0
    device = logits.device
    log_temperature = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).item())


def select_probability_threshold(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    if labels.numel() < 20 or logits.shape[1] <= 1:
        return 0.90
    probabilities = torch.softmax(logits / temperature, dim=1)
    pmax, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    selected: float | None = None
    for percentage in range(50, 100):
        candidate = percentage / 100.0
        accepted = pmax >= candidate
        if not accepted.any():
            continue
        precision = correct[accepted].float().mean().item()
        if precision >= 0.98:
            selected = candidate
            break
    return selected if selected is not None else 0.90


def build_prototypes(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    tag_to_id: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [C,256] prototypes and class-specific cosine gates in class-id order."""
    class_count = len(tag_to_id)
    prototypes: list[torch.Tensor] = []
    gates: list[float] = []
    for class_id in range(class_count):
        features = F.normalize(train_features[train_labels == class_id], dim=1)
        if not len(features):
            raise ProjectError(f"No train feature exists for class id {class_id}")
        prototypes.append(F.normalize(features.mean(dim=0), dim=0))
        count = len(features)
        if count == 1:
            gates.append(0.65)
        elif count < 5:
            gates.append(0.55)
        else:
            similarities: list[float] = []
            for index in range(count):
                others = torch.cat((features[:index], features[index + 1 :]), dim=0)
                loo_prototype = F.normalize(others.mean(dim=0), dim=0)
                similarities.append(float(torch.dot(features[index], loo_prototype).item()))
            gates.append(float(np.clip(np.quantile(similarities, 0.05), 0.45, 0.90)))
    return torch.stack(prototypes).cpu(), torch.tensor(gates, dtype=torch.float32)


def build_inference_artifacts(
    train: Evaluation,
    val: Evaluation | None,
    tag_to_id: Mapping[str, int],
) -> dict[str, object]:
    val_logits = val.logits if val is not None else train.logits.new_empty((0, train.logits.shape[1]))
    val_labels = val.labels if val is not None else train.labels.new_empty((0,), dtype=torch.long)
    temperature = calibrate_temperature(val_logits, val_labels)
    probability_threshold = select_probability_threshold(val_logits, val_labels, temperature)
    prototypes, similarity_thresholds = build_prototypes(train.features, train.labels, tag_to_id)
    return {
        "temperature": temperature,
        "probability_threshold": probability_threshold,
        "prototypes": prototypes,
        "similarity_thresholds": similarity_thresholds,
        "train_features": F.normalize(train.features, dim=1),
        "train_labels": train.labels,
    }


@dataclass(frozen=True)
class RejectedCandidate:
    probability: float
    similarity: float
    similarity_gate: float
    suggested_tag: str
    path: str
    feature: np.ndarray

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.probability + 0.5 * max(0.0, self.similarity_gate - self.similarity)


def active_learning_selection(
    rejected: Sequence[RejectedCandidate],
    labelled_features: torch.Tensor,
) -> list[RejectedCandidate]:
    """Select max 200 rejected images with uncertainty prefiltering and greedy k-center."""
    budget = min(200, len(rejected))
    if budget == 0:
        return []
    candidate_pool = sorted(rejected, key=lambda item: (-item.uncertainty, item.path))[: min(5 * budget, len(rejected))]
    features = np.stack([item.feature for item in candidate_pool]).astype(np.float32, copy=False)
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
    labelled = labelled_features.detach().cpu().numpy().astype(np.float32, copy=False)
    if len(labelled):
        labelled /= np.linalg.norm(labelled, axis=1, keepdims=True).clip(min=1e-12)
        min_distance = 1.0 - np.max(features @ labelled.T, axis=1)
    else:
        min_distance = np.full(len(candidate_pool), np.inf, dtype=np.float32)
    chosen_indices: list[int] = []
    for _ in range(budget):
        if not chosen_indices and not len(labelled):
            index = 0  # Candidate pool already has deterministic descending uncertainty order.
        else:
            index = int(np.argmax(min_distance))
        chosen_indices.append(index)
        min_distance[index] = -np.inf
        new_distance = 1.0 - features @ features[index]
        min_distance = np.minimum(min_distance, new_distance)
        min_distance[chosen_indices] = -np.inf
    return [candidate_pool[index] for index in chosen_indices]
