# src/image_auto_classifier/inferencer.py
"""Registered-model inference and rejection-gated copying.

Performance notes
-----------------
The previous implementation decoded every input image **once per registered
model**, which dominated wall time when ``__UNITEINFER__`` compared many
models. The current code decodes each image exactly once (``_decode_all``)
and reuses the CPU tensor cache across every model, so decode cost is paid
only once regardless of how many models are registered. Model execution is
staged: all registered checkpoints are loaded onto CPU, then each model is
moved to CUDA, runs its forward pass over the shared decoded cache, and is
moved back to CPU before the next model starts. This keeps the peak GPU
footprint to a single DINOv3 instance while removing the redundant decode
work.

Additional accelerations:
* TF32 and cuDNN auto-tuning are enabled for the CUDA session.
* Shared inference settings control batch size and worker counts. The
  ``decode_workers`` setting accepts ``0`` as a sentinel meaning "use all
  available CPU cores"; ``run_inference`` resolves the sentinel before
  calling ``_decode_all``.
* The eval transform is built once from the first registry's config and shared
  by every model.
* Host-to-device transfers are overlapped with compute through a dedicated
  ``torch.cuda.Stream`` (see ``_prefetched_batches``); on CPU hosts the
  helper degrades to a synchronous generator so no call site needs special
  casing.
* ``torch.cuda.empty_cache`` is *not* invoked between models; that call
  forces a device synchronize and only slows down the multi-model path.
  A single ``empty_cache`` runs after every model has finished.

``RegisteredModel`` is a frozen dataclass. Its ``prototypes`` and
``similarity_thresholds`` fields are therefore never mutated; device
placement is handled inside ``_run_one_model`` by binding local tensors.

OUTOFTAG semantics
------------------
The configured OUTOFTAG probability threshold is evaluated **per registered model**,
not just against the winning candidate. In ``__UNITEINFER__`` mode a single
image is compared against every registered checkpoint; if *any* of those
checkpoints assigns it a top-1 probability below the threshold the image is
treated as out-of-tag. Up to ``outoftag_copy_limit`` flagged images are copied
into ``output/OUTOFTAG/OUTOFTAG`` in ascending source-file size order. This
matches the single-dataset semantics, where the only participating model
decides OUTOFTAG membership, and prevents a confident prediction from one
model from silently hiding another model's uncertainty about the same image.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from .checkpointing import load_checkpoint, registered_model_paths, validate_metadata
from .common import ProjectError, is_allowed_image, require_cuda, sha256_file
from .config import InferenceConfig, load_inference_config, parse_config
from .data import _load_rgb_checked, build_transforms
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
    """Immutable bundle of one registered checkpoint and its inference artifacts.

    All tensor fields are frozen bindings. Consumers that need these tensors
    on a different device must create local bindings (e.g. ``x.to(device)``)
    rather than reassign the dataclass fields.
    """

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
        return DecodeFailure(
            path,
            path.relative_to(input_root).as_posix(),
            f"decode_error:{type(error).__name__}",
        )


def _enable_cuda_fast_paths(config: InferenceConfig) -> None:
    """Enable safe CUDA acceleration knobs for inference throughput.

    TF32 and cuDNN benchmark provide meaningful speed-ups on Ampere+ GPUs
    without changing numerical results beyond the TF32 mantissa, which is
    the standard inference trade-off. Both flags are no-ops on CPU-only
    hosts because the underlying routines never execute.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark


def _prefetched_batches(
    items: Sequence[PreparedImage],
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, list[PreparedImage]]]:
    """Yield CUDA-ready image batches, prefetching the next batch on a side stream.

    On CUDA the host-to-device copy of batch ``N+1`` is issued on a dedicated
    ``torch.cuda.Stream`` while batch ``N`` is being consumed by the model,
    hiding the transfer behind compute. On CPU (or when CUDA is unavailable)
    the generator degrades to a synchronous stack/transfer so the call site
    needs no special casing.

    Each batch allocates a fresh pinned CPU tensor, so no buffer is shared
    between iterations. The prefetch stream is therefore free to run ahead on
    memory the compute stream has not yet observed; a ``torch.cuda.Event``
    recorded immediately after the asynchronous copy is used to make the
    compute stream wait only when it actually needs the data. The pinned
    source tensor is kept alive until the next iteration so the pending
    asynchronous copy cannot outlive its backing storage.

    :param items: Prepared images in a deterministic order.
    :param batch_size: Maximum images per yield; must be positive.
    :param device: Destination device for the stacked tensor.
    :return: Iterator of ``(images_on_device, batch_items)`` tuples.
    :raises ProjectError: If ``batch_size`` is not a positive integer.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ProjectError("batch_size must be a positive integer")
    batches: list[list[PreparedImage]] = [
        list(items[start:start + batch_size]) for start in range(0, len(items), batch_size)
    ]
    if not batches:
        return

    if device.type != "cuda":
        for batch in batches:
            images = torch.stack([item.image for item in batch]).to(device)
            yield images, batch
        return

    prefetch_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)

    def prepare(batch: list[PreparedImage]) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        pinned = torch.stack([item.image for item in batch]).pin_memory()
        with torch.cuda.stream(prefetch_stream):
            images = pinned.to(device, non_blocking=True)
            event = torch.cuda.Event(enable_timing=False)
            event.record(prefetch_stream)
        return pinned, images, event

    next_prepared = prepare(batches[0])
    for index in range(len(batches)):
        upcoming = prepare(batches[index + 1]) if index + 1 < len(batches) else None
        # ``pinned`` remains bound until the next loop iteration assigns it
        # again, keeping the source buffer alive for the duration of the copy.
        pinned, images, event = next_prepared
        current_stream.wait_event(event)
        yield images, batches[index]
        next_prepared = upcoming


def load_registered_model(
    project_root: Path,
    dataset: str,
    device: torch.device,
    progress: Any | None = None,
) -> RegisteredModel:
    """Restore the registered model, placing weights and artifacts on ``device``.

    ``device`` may be ``torch.device("cpu")``, which is used by the
    multi-model inference path to keep several checkpoints resident in
    host memory before moving them to CUDA one at a time. The returned
    ``RegisteredModel`` is frozen; downstream code must not reassign its
    fields, and should instead bind device-local copies as needed.
    """
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

    report(f"[load] Moving model and inference artifacts to {device.type}...")
    model.to(device)
    model.eval()
    prototypes = prototypes.to(device=device, dtype=torch.float32)
    gates = gates.to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report("[load] Model is ready.")

    return RegisteredModel(
        model=model,
        checkpoint={"inference": artifacts, "config": config.to_dict()},
        metadata=metadata,
        id_to_tag=id_to_tag,
        prototypes=prototypes,
        similarity_thresholds=gates,
        temperature=temperature,
        probability_threshold=probability_threshold,
    )


def _decode_all(
    input_paths: Sequence[Path],
    input_root: Path,
    transform,  # type: ignore[no-untyped-def]
    *,
    max_workers: int,
) -> tuple[dict[str, PreparedImage], dict[str, DecodeFailure]]:
    """Decode every input image in parallel exactly once.

    Returns two dictionaries keyed by the input-relative POSIX path so the
    downstream inference loop can iterate deterministically in the original
    order without re-running the decoder for each registered model.

    ``max_workers`` must already be a concrete positive integer; the
    ``decode_workers == 0`` auto-detect sentinel is resolved by the caller
    before this helper is invoked.
    """
    from tqdm import tqdm

    prepared: dict[str, PreparedImage] = {}
    failures: dict[str, DecodeFailure] = {}
    if not input_paths:
        return prepared, failures
    if type(max_workers) is not int or max_workers < 1:
        raise ProjectError("decode max_workers must be a positive integer")
    workers = max(1, min(max_workers, len(input_paths)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-decode") as executor:
        futures = [executor.submit(_decode, path, input_root, transform) for path in input_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding", unit="image"):
            result = future.result()
            if isinstance(result, DecodeFailure):
                failures[result.relative_path] = result
            else:
                prepared[result.relative_path] = result
    return prepared, failures


def _run_one_model(
    *,
    registry: RegisteredModel,
    prepared: Mapping[str, PreparedImage],
    device: torch.device,
    threshold: float,
    config: InferenceConfig,
    current_dataset: str,
    best_candidates: dict[str, dict[str, object]],
    low_confidence_paths: set[str],
) -> None:
    """Run one model over the shared decoded cache and merge winning predictions.

    Images are processed in deterministic relative-path order using
    ``_prefetched_batches`` so the H2D copy of the next batch overlaps the
    forward pass of the current batch on CUDA hosts. The winning candidate
    per image is selected by the highest calibrated probability across every
    registered model.

    ``low_confidence_paths`` receives the relative path of every image whose
    top-1 calibrated probability under *this* model is below the configured
    OUTOFTAG probability threshold. The caller unions these sets across
    every registered model, so an image flagged by any model ends up in the
    OUTOFTAG bucket regardless of whether a different model happened to be
    more confident about it. This mirrors the single-dataset semantics where
    only the participating model can decide OUTOFTAG membership, and it fixes
    the previous behaviour in which a confident winner silently hid every
    other model's uncertainty about the same image.

    ``registry`` is a frozen dataclass; its ``prototypes`` and
    ``similarity_thresholds`` tensors are never reassigned. Device placement
    is achieved by binding local tensors once per call.
    """
    from tqdm import tqdm

    items = sorted(prepared.values(), key=lambda item: item.relative_path)
    if not items:
        return

    # Bind device-local views of the frozen inference artifacts once per model.
    # ``Tensor.to`` is a no-op when the tensor already lives on ``device``,
    # so the single-model GPU path pays nothing extra.
    prototypes = registry.prototypes.to(device=device, dtype=torch.float32)
    gates = registry.similarity_thresholds.to(device=device, dtype=torch.float32)

    with tqdm(total=len(items), desc=f"Inference [{current_dataset}]", unit="image") as progress:
        for images, batch in _prefetched_batches(items, config.batch_size, device):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=config.mixed_precision
            ):
                features, logits = registry.model(images, labels=None)
                probabilities = torch.softmax(logits / registry.temperature, dim=1)
                pmax, class_ids = probabilities.max(dim=1)
                similarities = (features * prototypes[class_ids]).sum(dim=1)
                class_gates = gates[class_ids]

            for item, probability, similarity, class_id, gate in zip(
                batch,
                pmax.float().cpu(),
                similarities.float().cpu(),
                class_ids.cpu(),
                class_gates.float().cpu(),
                strict=True,
            ):
                probability_value = float(probability.item())
                # Record per-model out-of-tag membership. Every registered
                # model votes independently, so an image whose top-1
                # probability falls below the threshold in *any* model is
                # surfaced for human review even when a different model is
                # more confident about it.
                if probability_value < config.outoftag_probability_threshold:
                    low_confidence_paths.add(item.relative_path)
                candidate = {
                    "dataset": current_dataset,
                    "path": item.path,
                    "tag": registry.id_to_tag[int(class_id.item())],
                    "probability": probability_value,
                    "similarity": float(similarity.item()),
                    "gate": float(gate.item()),
                    "threshold": threshold,
                }
                previous = best_candidates.get(item.relative_path)
                if previous is None or probability_value > float(previous["probability"]):
                    best_candidates[item.relative_path] = candidate
            progress.update(len(batch))


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


def _select_outoftag_paths(input_dir: Path, flagged_paths: set[str], limit: int) -> list[Path]:
    """Choose flagged input files by ascending byte size, then relative path.

    :raises ProjectError: If a flagged source cannot be sized reliably.
    """
    if limit == 0:
        return []
    sized_paths: list[tuple[int, str, Path]] = []
    for relative_path in flagged_paths:
        source = input_dir / relative_path
        try:
            size = source.stat().st_size
        except OSError as error:
            raise ProjectError(f"Cannot determine OUTOFTAG source size: {source}: {error}") from error
        sized_paths.append((size, relative_path, source))
    sized_paths.sort()
    return [source for _, _, source in sized_paths[:limit]]


def _resolve_decode_workers(config: InferenceConfig) -> int:
    """Resolve the ``decode_workers`` auto-detect sentinel to a concrete count.

    A configured value of ``0`` means "use every logical CPU core", with a
    conservative fallback of ``4`` on platforms where ``os.cpu_count()``
    returns ``None`` or zero. Any positive value is honoured verbatim.
    """
    if config.decode_workers > 0:
        return config.decode_workers
    detected = os.cpu_count() or 0
    return max(1, detected) if detected else 4


def run_inference(
    *,
    project_root: Path,
    dataset: str | None,
    input_dir: Path,
    probability_threshold: float | None,
    inference_config_path: Path | None = None,
) -> tuple[int, int]:
    """Classify input images with one model or every registered dataset model.

    Passing ``None`` or ``__UNITEINFER__`` for ``dataset`` compares the maximum
    class probability from every registered dataset model for each image. The
    winning model's rejection gates still determine whether that image is copied.

    OUTOFTAG membership is decided per registered model: any image whose top-1
    probability under *some* model falls below the shared inference setting
    is copied into ``output/OUTOFTAG/OUTOFTAG``, regardless of how confident a
    different model is about that image. This matches the single-dataset path,
    where the only participating model makes the OUTOFTAG decision on its own.
    All flagged images are recorded before selecting the smallest source files
    for the configured OUTOFTAG copy limit. Equal-sized files use relative-path
    order, so the selection is deterministic.

    Performance contract: decode work is paid exactly once per input image
    (not once per registered model), each model is moved to CUDA for the
    duration of its forward pass only, H2D transfers are prefetched on a side
    stream, and ``torch.cuda.empty_cache`` runs once after the model loop
    instead of between models. Returns accepted and rejected image counts.
    """
    from tqdm import tqdm

    config_path = inference_config_path or project_root / "configs" / "inference.yaml"
    try:
        inference_config = load_inference_config(config_path)
    except (OSError, ValueError) as error:
        raise ProjectError(f"Cannot load inference configuration {config_path}: {error}") from error
    device = require_cuda()
    _enable_cuda_fast_paths(inference_config)

    if not input_dir.is_dir():
        raise ProjectError(f"Input directory does not exist: {input_dir}")
    if (probability_threshold is not None
            and (not math.isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0)):
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
    if not input_paths:
        return 0, 0

    # Stage 1: load every registered checkpoint onto CPU. Keeping the weights
    # on the host avoids reserving N copies of the DINOv3 graph in VRAM while
    # still letting us overlap the checkpoint I/O across models.
    tqdm.write(f"[load] Loading {len(datasets)} registered model(s) onto CPU...")
    registries: list[tuple[str, RegisteredModel]] = []
    for name in datasets:
        registry = load_registered_model(project_root, name, torch.device("cpu"), progress=tqdm.write)
        registries.append((name, registry))

    # Stage 2: single shared eval transform, derived from the first registry.
    first_config = parse_config(registries[0][1].checkpoint["config"])
    _, eval_transform = build_transforms(first_config.data.input_size, config=first_config.data)

    # Stage 3: decode every input image exactly once and cache it on CPU.
    decode_workers = _resolve_decode_workers(inference_config)
    tqdm.write(
        f"[decode] Decoding {len(input_paths)} image(s) exactly once with {decode_workers} worker(s)..."
    )
    prepared, failures = _decode_all(
        input_paths, input_dir, eval_transform, max_workers=decode_workers
    )
    if not prepared:
        tqdm.write("[inference] No decodable images.")

    # Stage 4: run each model over the shared cache. Models are moved to CUDA
    # one at a time and released before the next model is scheduled.
    best_candidates: dict[str, dict[str, object]] = {}
    # Union of every image flagged as out-of-tag by any registered model.
    # This is deliberately collected per model in ``_run_one_model`` instead
    # of being derived from ``best_candidates``, because ``best_candidates``
    # only keeps the single most confident model per image and would otherwise
    # erase every other model's OUTOFTAG vote.
    low_confidence_paths: set[str] = set()
    for current_dataset, registry in registries:
        tqdm.write(f"[load] Preparing CUDA inference for dataset '{current_dataset}'...")
        threshold = registry.probability_threshold if probability_threshold is None else probability_threshold

        # ``nn.Module.to`` is an in-place method; it does not reassign any
        # field of the frozen ``RegisteredModel`` dataclass.
        registry.model.to(device)
        registry.model.eval()
        try:
            _run_one_model(
                registry=registry,
                prepared=prepared,
                device=device,
                threshold=threshold,
                config=inference_config,
                current_dataset=current_dataset,
                best_candidates=best_candidates,
                low_confidence_paths=low_confidence_paths,
            )
        finally:
            # Release the CUDA allocations before loading the next model.
            # ``empty_cache`` is intentionally *not* called here: it forces a
            # device synchronize and prevents the caching allocator from
            # reusing the freed blocks for the next model. A single call at
            # the end of the loop reclaims any residual cached blocks.
            registry.model.to("cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 5: copy accepted images and select OUTOFTAG copies by file size.
    accepted_count = 0
    rejected_count = 0
    copy_futures = []
    outoftag_dir = project_root / "output" / "OUTOFTAG" / "OUTOFTAG"
    outoftag_paths = _select_outoftag_paths(
        input_dir, low_confidence_paths, inference_config.outoftag_copy_limit
    )
    tqdm.write(f"[OUTOFTAG] flagged={len(low_confidence_paths)} copying={len(outoftag_paths)}")

    with ThreadPoolExecutor(max_workers=inference_config.copy_workers, thread_name_prefix="image-copy") as copy_executor:
        for path in input_paths:
            relative_path = path.relative_to(input_dir).as_posix()
            failure = failures.get(relative_path)
            candidate = best_candidates.get(relative_path)
            if failure is not None or candidate is None:
                rejected_count += 1
                continue

            predicted_dataset = str(candidate["dataset"])
            predicted_tag = str(candidate["tag"])
            probability_value = float(candidate["probability"])
            similarity_value = float(candidate["similarity"])
            gate_value = float(candidate["gate"])
            threshold = float(candidate["threshold"])

            accepted = probability_value >= threshold and similarity_value >= gate_value
            if accepted:
                accepted_count += 1
                copy_futures.append(
                    copy_executor.submit(
                        _copy_accepted,
                        candidate["path"],
                        project_root / "output" / predicted_dataset / predicted_tag,
                    )
                )
            else:
                rejected_count += 1

        for source in outoftag_paths:
            copy_futures.append(copy_executor.submit(_copy_accepted, source, outoftag_dir))

        for future in copy_futures:
            future.result()

    return accepted_count, rejected_count
