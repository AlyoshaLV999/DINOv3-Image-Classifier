"""EMA evaluation, calibration, and class prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import ProjectError
from .model import MetricArcFaceModel, ParameterEMA


@dataclass(frozen=True)
class Evaluation:
    """Snapshot of one evaluation split.

    ``macro_f1``, ``weighted_f1``, and ``balanced_accuracy`` are all computed
    only over classes that actually appear in the evaluated split. Classes
    that are absent from the split are excluded so that a class which has no
    validation/test images can never silently drag the aggregate metrics down
    (sklearn's default macro average would otherwise include any class the
    model predicts, even if it is absent from ``y_true``).

    ``weighted_f1`` is the support-weighted counterpart of ``macro_f1`` and
    gives the sample-weighted view, which is dominated by large classes.
    ``balanced_accuracy`` is the unweighted mean of per-class recall and
    therefore stays comparable across very different class support.
    ``evaluated_class_count`` records how many classes actually contributed
    to the aggregate metrics.
    """

    loss: float
    top1: float
    precision: float
    recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    evaluated_class_count: int
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
    description: str = "eval",
) -> Evaluation:
    """Evaluate EMA with no augmentation or margin, restoring model state on errors.

    Train and validation must use loaders with the same deterministic transform.
    Retains the existing sklearn macro-metric convention for checkpoint compatibility.

    Aggregate metrics are restricted to the set of classes that actually appear
    in the evaluated split. ``macro_f1`` therefore reports the mean F1 over
    present classes only, ``weighted_f1`` complements it with the sample-weighted
    view dominated by large classes, and ``balanced_accuracy`` uses the
    unweighted mean of per-class recall so it remains stable when the split is
    highly imbalanced. Classes that are absent from the split are excluded
    entirely so they can never be silently scored as zero.

    ``description`` labels an optional ``tqdm`` progress bar. The previous silent
    evaluation loop accounted for a large share of the visible "stall" between
    the training progress bar finishing and the next epoch starting; making the
    loop observable costs nothing measurable while allowing operators to
    distinguish evaluation time from checkpoint write time. The bar is only
    rendered when ``tqdm`` decides stdout is a suitable target, so redirected
    logs are unaffected.

    :param model: Model whose EMA parameters will be evaluated.
    :param ema: ParameterEMA instance supplying the averaged weights.
    :param loader: DataLoader yielding ``(images, labels)`` batches.
    :param device: Device the model and batches are placed on.
    :param amp_enabled: Whether to run the forward pass under FP16 autocast.
    :param description: Short label used for the progress bar.
    :return: An ``Evaluation`` snapshot with detached CPU tensors.
    :raises ProjectError: If the loader yielded no batches, or if the split
        contains no labelled samples.
    """
    from tqdm import tqdm

    was_training = model.training
    model.eval()
    features_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    try:
        with ema.average_parameters(model), torch.inference_mode():
            progress = tqdm(loader, desc=description, unit="batch", leave=False)
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    features, logits = model(images, labels=None)
                features_parts.append(features.float().cpu())
                logits_parts.append(logits.float().cpu())
                labels_parts.append(labels.cpu())
    finally:
        model.train(was_training)
    if not labels_parts:
        raise ProjectError("Evaluation loader produced no batches")
    features = torch.cat(features_parts)
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    loss = F.cross_entropy(logits, labels).item() if logits.shape[1] > 1 else 0.0
    predictions = logits.argmax(dim=1)
    targets_np = labels.numpy()
    predictions_np = predictions.numpy()

    # Restrict aggregate metrics to classes that actually have support in this
    # split. Sklearn's default ``labels=None`` uses the sorted union of y_true
    # and y_pred; a class that the model happens to predict but that is absent
    # from the split would silently contribute precision=recall=f1=0 and pull
    # the macro average down. Pinning the label set makes the metric
    # deterministic and comparable across epochs and splits.
    evaluated_class_ids = np.unique(targets_np)
    if evaluated_class_ids.size == 0:
        raise ProjectError("Evaluation split contains no labelled samples")

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        targets_np,
        predictions_np,
        labels=evaluated_class_ids,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets_np,
        predictions_np,
        labels=evaluated_class_ids,
        average="weighted",
        zero_division=0,
    )

    # Balanced accuracy is the unweighted mean of per-class recall. Compute it
    # explicitly over ``evaluated_class_ids`` so classes absent from this split
    # cannot silently contribute a zero recall, and so the metric stays
    # consistent with the macro metrics reported above.
    class_recalls = [
        float(np.mean(predictions_np[targets_np == class_id] == class_id))
        for class_id in evaluated_class_ids
    ]
    balanced_accuracy = float(np.mean(class_recalls)) if class_recalls else 0.0

    return Evaluation(
        loss=float(loss),
        top1=float((predictions == labels).float().mean().item()),
        precision=float(precision),
        recall=float(recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        balanced_accuracy=balanced_accuracy,
        evaluated_class_count=int(evaluated_class_ids.size),
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
            # The leave-one-out mean is (sum - current) / (count - 1).
            # The positive divisor cancels during normalization.
            others = F.normalize(features.sum(dim=0, keepdim=True) - features, dim=1)
            similarities = (features * others).sum(dim=1).detach().cpu().numpy()
            gates.append(float(np.clip(np.quantile(similarities, 0.05), 0.45, 0.90)))
    return torch.stack(prototypes).cpu(), torch.tensor(gates, dtype=torch.float32)


def build_inference_artifacts(
    train: Evaluation,
    val: Evaluation | None,
    tag_to_id: Mapping[str, int],
) -> dict[str, object]:
    """Build train-only prototypes and calibrate on the supplied held-out evaluation.

    ``val`` retains its legacy parameter name; callers may supply an independent
    calibration split. Never supply test data. None uses conservative defaults.
    """
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
    }
