"""Registered-model inference, rejection-gated copying, and active-learning export."""

from __future__ import annotations

import csv
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .checkpointing import load_checkpoint, registered_model_paths, validate_metadata
from .common import ProjectError, is_allowed_image, require_cuda, sha256_file, utc_timestamp
from .config import parse_config
from .data import _load_rgb_checked, build_transforms
from .metrics import RejectedCandidate, active_learning_selection
from .model import MetricArcFaceModel, ParameterEMA


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    relative_path: str
    image: torch.Tensor


@dataclass(frozen=True)
class DecodeFailure:
    path: Path
    relative_path: str
    reason: str


@dataclass(frozen=True)
class RegisteredModel:
    model: MetricArcFaceModel
    checkpoint: Mapping[str, Any]
    metadata: Mapping[str, Any]
    id_to_tag: tuple[str, ...]
    prototypes: torch.Tensor
    similarity_thresholds: torch.Tensor
    temperature: float
    probability_threshold: float


def _decode(path: Path, input_root: Path, transform) -> PreparedImage | DecodeFailure:  # type: ignore[no-untyped-def]
    try:
        image = _load_rgb_checked(path)
        return PreparedImage(path, path.relative_to(input_root).as_posix(), transform(image))
    except Exception as error:
        return DecodeFailure(path, path.relative_to(input_root).as_posix(), f"decode_error:{type(error).__name__}")


def load_registered_model(
    project_root: Path,
    dataset: str,
    device: torch.device,
    progress: Any | None = None,
) -> RegisteredModel:
    """Restore the registered model on CPU, then transfer it to CUDA once."""
    report = progress if progress is not None else lambda _message: None
    if not dataset or dataset in {".", ".."} or "/" in dataset or "\\" in dataset:
        raise ProjectError("--dataset must be a single directory name")
    registry_dir = project_root / "models" / dataset
    checkpoint_path, metadata_path = registered_model_paths(registry_dir)

    report("[load] Reading registered checkpoint into CPU memory...")
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    report("[load] Validating checkpoint metadata...")
    metadata = validate_metadata(
        metadata_path,
        checkpoint,
        dataset,
        registry_version=checkpoint_path.parent.name,
    )
    try:
        config = parse_config(checkpoint["config"])
    except Exception as error:
        raise ProjectError("Registered checkpoint has an invalid configuration") from error

    tag_to_id = checkpoint["tag_to_id"]
    if not isinstance(tag_to_id, Mapping):
        raise ProjectError("Registered checkpoint tag mapping is invalid")
    id_to_tag = tuple(tag for tag, _ in sorted(tag_to_id.items(), key=lambda pair: pair[1]))
    if sorted(tag_to_id.values()) != list(range(len(id_to_tag))):
        raise ProjectError("Registered checkpoint tag ids must be contiguous integers beginning at zero")
    if config.dataset != dataset:
        raise ProjectError("Registered checkpoint configuration dataset does not match --dataset")
    if tuple(metadata.get("tags", [])) != id_to_tag:
        raise ProjectError("Registered metadata tag order does not match checkpoint")

    report("[load] Building the DINOv3 topology and applying trained weights...")
    model = MetricArcFaceModel(
        config.model,
        len(id_to_tag),
        local_files_only=True,
        load_pretrained_weights=False,
    )
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception as error:
        raise ProjectError(f"Registered checkpoint model weights are invalid: {error}") from error

    ema = ParameterEMA(model)
    ema.load_state_dict(checkpoint["ema_state"], model)
    ema.copy_to(model)

    artifacts = checkpoint["inference"]
    if not isinstance(artifacts, Mapping):
        raise ProjectError("Registered checkpoint inference artifacts are invalid")
    prototypes = artifacts.get("prototypes")
    gates = artifacts.get("similarity_thresholds")
    if not isinstance(prototypes, torch.Tensor) or not isinstance(gates, torch.Tensor):
        raise ProjectError("Registered checkpoint lacks prototype rejection artifacts")
    if prototypes.shape != (len(id_to_tag), 256) or gates.shape != (len(id_to_tag),):
        raise ProjectError("Registered checkpoint prototype artifact shape is invalid")

    temperature = float(artifacts.get("temperature", 0.0))
    probability_threshold = float(artifacts.get("probability_threshold", -1.0))
    if temperature <= 0.0 or not 0.0 <= probability_threshold <= 1.0:
        raise ProjectError("Registered checkpoint calibration artifacts are invalid")

    report("[load] Moving model and inference artifacts to CUDA...")
    model.to(device)
    model.eval()
    prototypes = prototypes.to(device=device, dtype=torch.float32)
    gates = gates.to(device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    report("[load] Model is ready.")

    return RegisteredModel(
        model=model,
        checkpoint={"inference": artifacts},
        metadata=metadata,
        id_to_tag=id_to_tag,
        prototypes=prototypes,
        similarity_thresholds=gates,
        temperature=temperature,
        probability_threshold=probability_threshold,
    )


def _copy_accepted(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        digest = sha256_file(source)[:8]
        destination = destination_dir / f"{source.stem}__{digest}{source.suffix}"
        suffix = 2
        while destination.exists():
            destination = destination_dir / f"{source.stem}__{digest}_{suffix}{source.suffix}"
            suffix += 1
    import shutil

    shutil.copy2(source, destination)
    return destination


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["relative_path", "predicted_tag", "probability", "cosine_similarity", "accepted", "rejection_reason"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _write_active_learning(
    *,
    project_root: Path,
    task_name: str,
    rejected: Sequence[RejectedCandidate],
    labelled_features: torch.Tensor,
) -> Path | None:
    selected = active_learning_selection(rejected, labelled_features)
    if not selected:
        return None
    path = project_root / "logs" / task_name / "active_learning" / f"{utc_timestamp()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "u", "pmax", "similarity", "suggested_tag", "path"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, item in enumerate(selected, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "u": f"{item.uncertainty:.8f}",
                        "pmax": f"{item.probability:.8f}",
                        "similarity": f"{item.similarity:.8f}",
                        "suggested_tag": item.suggested_tag,
                        "path": item.path,
                    }
                )
            handle.flush()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
    return path


def run_inference(
    *,
    project_root: Path,
    dataset: str | None,
    input_dir: Path,
    probability_threshold: float | None,
) -> tuple[int, int, tuple[Path, ...]]:
    """Classify input images with one model or every registered dataset model.

    Passing ``None`` or ``__UNITEINFER__`` for ``dataset`` compares the maximum
    class probability from every registered dataset model for each image. The
    winning model's rejection gates still determine whether that image is copied.
    Models are loaded and released one at a time to avoid retaining multiple
    DINOv3 instances on the GPU.
    """
    from tqdm import tqdm

    device = require_cuda()
    if not input_dir.is_dir():
        raise ProjectError(f"Input directory does not exist: {input_dir}")
    if probability_threshold is not None and not 0.0 <= probability_threshold <= 1.0:
        raise ProjectError("--threshold must be between 0 and 1")

    united = dataset is None or dataset == "__UNITEINFER__"
    if united:
        models_dir = project_root / "models"
        if not models_dir.is_dir():
            raise ProjectError(f"Model registry directory does not exist: {models_dir}")
        datasets = sorted(
            path.name for path in models_dir.iterdir() if path.is_dir() and (path / "current.json").is_file()
        )
        if not datasets:
            raise ProjectError("No registered dataset models were found for __UNITEINFER__")
    else:
        if dataset is None:
            raise ProjectError("--dataset must be a single directory name or __UNITEINFER__")
        datasets = [dataset]

    input_paths = sorted((path for path in input_dir.iterdir() if is_allowed_image(path)), key=lambda path: path.name)
    best_candidates: dict[str, dict[str, object]] = {}
    failures: dict[str, DecodeFailure] = {}
    active_learning_inputs: dict[str, tuple[str, torch.Tensor]] = {}

    for current_dataset in datasets:
        tqdm.write(f"[load] Preparing CUDA inference for dataset '{current_dataset}'...")
        registry = load_registered_model(project_root, current_dataset, device, progress=tqdm.write)
        threshold = registry.probability_threshold if probability_threshold is None else probability_threshold
        _, eval_transform = build_transforms(int(registry.metadata["preprocessing"]["input_size"]))
        train_features = registry.checkpoint["inference"].get("train_features")
        if not isinstance(train_features, torch.Tensor):
            raise ProjectError("Registered checkpoint lacks labelled features for active learning")
        active_learning_inputs[current_dataset] = (str(registry.metadata["task_name"]), train_features)

        tqdm.write(f"[inference] Running dataset '{current_dataset}' over {len(input_paths)} images...")
        batch: list[PreparedImage] = []
        with (
            ThreadPoolExecutor(max_workers=4, thread_name_prefix="image-decode") as decode_executor,
            tqdm(total=len(input_paths), desc=f"Inference [{current_dataset}]", unit="image") as progress,
        ):
            decoded = decode_executor.map(lambda path: _decode(path, input_dir, eval_transform), input_paths)
            for _ in range(len(input_paths) + 1):
                result = next(decoded, None)
                if result is not None:
                    progress.update(1)
                    if isinstance(result, DecodeFailure):
                        failures.setdefault(result.relative_path, result)
                        continue
                    batch.append(result)

                if len(batch) < 32 and result is not None:
                    continue
                if not batch:
                    continue

                images = torch.stack([item.image for item in batch]).pin_memory().to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    features, logits = registry.model(images, labels=None)
                    probabilities = torch.softmax(logits / registry.temperature, dim=1)
                    pmax, class_ids = probabilities.max(dim=1)
                    similarities = (features * registry.prototypes[class_ids]).sum(dim=1)
                    gates = registry.similarity_thresholds[class_ids]

                for item, feature, probability, similarity, class_id, gate in zip(
                    batch,
                    features.float().cpu(),
                    pmax.float().cpu(),
                    similarities.float().cpu(),
                    class_ids.cpu(),
                    gates.float().cpu(),
                    strict=True,
                ):
                    probability_value = float(probability.item())
                    candidate = {
                        "dataset": current_dataset,
                        "path": item.path,
                        "tag": registry.id_to_tag[int(class_id.item())],
                        "probability": probability_value,
                        "similarity": float(similarity.item()),
                        "gate": float(gate.item()),
                        "threshold": threshold,
                        "feature": feature,
                    }
                    previous = best_candidates.get(item.relative_path)
                    if previous is None or probability_value > float(previous["probability"]):
                        best_candidates[item.relative_path] = candidate
                batch.clear()

        del registry
        torch.cuda.empty_cache()

    rows: list[dict[str, object]] = []
    rejected_by_dataset: dict[str, list[RejectedCandidate]] = {current_dataset: [] for current_dataset in datasets}
    copy_futures = []
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="image-copy") as copy_executor:
        for path in input_paths:
            relative_path = path.relative_to(input_dir).as_posix()
            failure = failures.get(relative_path)
            candidate = best_candidates.get(relative_path)
            if failure is not None or candidate is None:
                rows.append(
                    {
                        "relative_path": relative_path,
                        "predicted_tag": "",
                        "probability": "",
                        "cosine_similarity": "",
                        "accepted": False,
                        "rejection_reason": failure.reason if failure is not None else "no_prediction",
                    }
                )
                continue

            predicted_dataset = str(candidate["dataset"])
            predicted_tag = str(candidate["tag"])
            probability_value = float(candidate["probability"])
            similarity_value = float(candidate["similarity"])
            gate_value = float(candidate["gate"])
            threshold = float(candidate["threshold"])
            reasons: list[str] = []
            if probability_value < threshold:
                reasons.append("probability_below_threshold")
            if similarity_value < gate_value:
                reasons.append("similarity_below_class_threshold")

            accepted = not reasons
            if accepted:
                copy_futures.append(
                    copy_executor.submit(
                        _copy_accepted,
                        candidate["path"],
                        project_root / "output" / predicted_dataset / predicted_tag,
                    )
                )
            else:
                feature = candidate["feature"]
                if not isinstance(feature, torch.Tensor):
                    raise ProjectError("Inference feature has an invalid type")
                rejected_by_dataset[predicted_dataset].append(
                    RejectedCandidate(
                        probability=probability_value,
                        similarity=similarity_value,
                        similarity_gate=gate_value,
                        suggested_tag=predicted_tag,
                        path=str(Path(candidate["path"]).resolve()),
                        feature=feature.numpy(),
                    )
                )

            rows.append(
                {
                    "relative_path": relative_path,
                    "predicted_tag": predicted_tag,
                    "probability": f"{probability_value:.8f}",
                    "cosine_similarity": f"{similarity_value:.8f}",
                    "accepted": accepted,
                    "rejection_reason": ";".join(reasons),
                }
            )

        for future in copy_futures:
            future.result()

    active_paths: list[Path] = []
    for current_dataset, rejected in rejected_by_dataset.items():
        task_name, train_features = active_learning_inputs[current_dataset]
        active_path = _write_active_learning(
            project_root=project_root,
            task_name=task_name,
            rejected=rejected,
            labelled_features=train_features,
        )
        if active_path is not None:
            active_paths.append(active_path)

    return sum(bool(row["accepted"]) for row in rows), sum(not bool(row["accepted"]) for row in rows), tuple(active_paths)
