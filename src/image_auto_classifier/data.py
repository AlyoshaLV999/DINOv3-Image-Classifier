"""Dataset audit, immutable splits, caching, transforms, and balanced sampling."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from .config import DataConfig

from .common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ProjectError,
    atomic_write_text,
    is_allowed_image,
    sha256_file,
    stable_json_dumps,
)


MAX_PIXELS = 250_000_000
register_heif_opener()


@dataclass(frozen=True)
class ManifestItem:
    source_rel_path: str
    cache_rel_path: str
    sha256: str
    tag: str
    split: str

    @classmethod
    def from_dict(cls, value: Mapping[str, str]) -> "ManifestItem":
        expected = {"source_rel_path", "cache_rel_path", "sha256", "tag", "split"}
        if set(value) != expected:
            raise ProjectError("split_manifest.jsonl contains an unsupported record")
        item = cls(**value)
        if item.split not in {"train", "val", "calibration", "test"}:
            raise ProjectError("split_manifest.jsonl contains an invalid split")
        return item


@dataclass(frozen=True)
class AuditedImage:
    source_path: Path
    source_rel_path: str
    cache_rel_path: str
    sha256: str
    tag: str


def _load_rgb_checked(path: Path, *, draft_long_side: int | None = None) -> Image.Image:
    """Decode one image, optionally downsampling JPEG data before full decode."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("animated or multi-frame images are not supported")
        width, height = opened.size
        if width <= 0 or height <= 0:
            raise ValueError("image has zero dimension")
        if width * height > MAX_PIXELS:
            raise OverflowError(f"oversize image ({width}x{height}, limit {MAX_PIXELS} pixels)")
        if draft_long_side is not None:
            if draft_long_side <= 0:
                raise ValueError("draft_long_side must be positive")
            if opened.format == "JPEG" and max(width, height) > draft_long_side:
                scale = draft_long_side / max(width, height)
                opened.draft(
                    "RGB",
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                )
        image = ImageOps.exif_transpose(opened)
        image.load()
        return image.convert("RGB")


def _cache_image(image: Image.Image, cache_path: Path, long_side: int) -> None:
    if cache_path.exists():
        return
    width, height = image.size
    scale = min(1.0, long_side / max(width, height))
    if scale < 1.0:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.webp")
    try:
        image.save(temporary, format="WEBP", lossless=True, method=6)
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> list[ManifestItem]:
    if not path.exists():
        return []
    items: list[ManifestItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                items.append(ManifestItem.from_dict(raw))
            except (json.JSONDecodeError, TypeError, ProjectError) as error:
                raise ProjectError(f"Invalid split manifest at line {line_no}: {error}") from error
    return items


def _write_manifest(path: Path, items: Sequence[ManifestItem]) -> None:
    body = "".join(stable_json_dumps(asdict(item)) + "\n" for item in items)
    atomic_write_text(path, body)


def audit_dataset(
    *,
    project_root: Path,
    dataset: str,
    task_log_dir: Path,
    cache_long_side: int,
    logger: logging.Logger,
    create_split_manifest: bool = True,
    split_config: DataConfig = DataConfig(),
) -> tuple[list[ManifestItem], list[str]]:
    """Synchronize source images into a durable, tag-aware cache before splitting.

    ``cache/<dataset>/cache_index.jsonl`` is the dataset snapshot: it stores each
    image's tag and cache path independently of a task. Consequently a new task can
    reuse the complete cache even when ``datasets/<dataset>`` has been removed. When
    the source directory is present, newly discovered files are hashed and missing
    cache entries are decoded and encoded concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    from tqdm import tqdm

    dataset_root = project_root / "datasets" / dataset
    cache_root = project_root / "cache" / dataset
    cache_images_root = cache_root / "images"
    cache_index_path = cache_root / "cache_index.jsonl"
    cache_index_fields = {"source_rel_path", "cache_rel_path", "sha256", "tag"}

    def is_safe_relative_path(value: str) -> bool:
        path = Path(value)
        return bool(value) and "\\" not in value and not path.is_absolute() and ".." not in path.parts

    records_by_key: dict[tuple[str, str], AuditedImage] = {}
    tag_by_digest: dict[str, str] = {}
    index_needs_write = not cache_index_path.exists()

    def add_cache_record(record: AuditedImage, *, origin: str) -> None:
        key = (record.tag, record.sha256)
        previous_tag = tag_by_digest.get(record.sha256)
        if previous_tag is not None and previous_tag != record.tag:
            raise ProjectError(
                f"Cross-tag duplicate SHA-256 conflict in {origin}: {record.sha256} "
                f"belongs to both '{previous_tag}' and '{record.tag}'"
            )
        tag_by_digest[record.sha256] = record.tag
        records_by_key.setdefault(key, record)

    if cache_index_path.exists():
        with cache_index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or set(raw) != cache_index_fields:
                        raise ProjectError("record has an unsupported schema")
                    if not all(isinstance(raw[field], str) for field in cache_index_fields):
                        raise ProjectError("record fields must be strings")
                    source_rel_path = raw["source_rel_path"]
                    cache_rel_path = raw["cache_rel_path"]
                    digest = raw["sha256"]
                    tag = raw["tag"]
                    if (
                        not is_safe_relative_path(source_rel_path)
                        or not is_safe_relative_path(cache_rel_path)
                        or not tag
                        or tag in {".", ".."}
                        or tag != Path(tag).name
                        or "\\" in tag
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ProjectError("record contains an invalid path, tag, or SHA-256")
                    cache_path = project_root / cache_rel_path
                    try:
                        cache_path.relative_to(cache_root)
                    except ValueError as error:
                        raise ProjectError("record cache path is outside this dataset cache") from error
                    if cache_path.name != f"{digest}.webp":
                        raise ProjectError("record cache filename does not match its SHA-256")
                    add_cache_record(
                        AuditedImage(dataset_root / source_rel_path, source_rel_path, cache_rel_path, digest, tag),
                        origin=f"cache index line {line_no}",
                    )
                except (json.JSONDecodeError, TypeError, ProjectError) as error:
                    raise ProjectError(f"Invalid cache index at line {line_no}: {error}") from error
    else:
        # Versions before the durable index kept tag metadata only in task manifests.
        # Import those records once so users can upgrade without losing an existing cache.
        logs_root = project_root / "logs"
        migrated_records = 0
        if logs_root.is_dir():
            for manifest_path in sorted(logs_root.glob("*/split_manifest.jsonl"), key=lambda path: path.parent.name):
                for item in _read_manifest(manifest_path):
                    cache_path = project_root / item.cache_rel_path
                    try:
                        cache_path.relative_to(cache_root)
                    except ValueError:
                        continue
                    if not cache_path.is_file():
                        continue
                    if (
                        not is_safe_relative_path(item.source_rel_path)
                        or not is_safe_relative_path(item.cache_rel_path)
                        or not item.tag
                        or item.tag in {".", ".."}
                        or item.tag != Path(item.tag).name
                        or "\\" in item.tag
                        or len(item.sha256) != 64
                        or any(character not in "0123456789abcdef" for character in item.sha256)
                        or cache_path.name != f"{item.sha256}.webp"
                    ):
                        logger.warning("Skipping invalid legacy cache record from %s", manifest_path)
                        continue
                    before = len(records_by_key)
                    add_cache_record(
                        AuditedImage(
                            dataset_root / item.source_rel_path,
                            item.source_rel_path,
                            item.cache_rel_path,
                            item.sha256,
                            item.tag,
                        ),
                        origin=str(manifest_path),
                    )
                    migrated_records += len(records_by_key) - before
        if migrated_records:
            logger.info("Migrated %d cached images into %s", migrated_records, cache_index_path)

    active_by_key: dict[tuple[str, str], AuditedImage] = {}
    for key, record in records_by_key.items():
        if (project_root / record.cache_rel_path).is_file():
            active_by_key[key] = record
        else:
            logger.warning(
                "Cache file is missing and will be rebuilt if its source is available: %s",
                record.cache_rel_path,
            )
            index_needs_write = True

    declared_tags: set[str] = set()
    source_entries: list[tuple[str, Path]] = []
    if dataset_root.is_dir():
        tag_dirs = sorted(
            (path for path in dataset_root.iterdir() if path.is_dir() and any(path.iterdir())),
            key=lambda path: path.name,
        )
        for tag_dir in tag_dirs:
            tag = tag_dir.name
            declared_tags.add(tag)
            files = sorted((path for path in tag_dir.iterdir() if is_allowed_image(path)), key=lambda path: path.name)
            if not files:
                logger.warning("Tag '%s' has no directly contained supported image files", tag)
            source_entries.extend((tag, path) for path in files)
    elif records_by_key:
        logger.info("Dataset directory is absent; training from the durable cache index: %s", cache_index_path)

    worker_count = min(32, max(1, (os.cpu_count() or 1) * 2))
    hashes_by_source: dict[tuple[str, Path], str] = {}
    if source_entries:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cache-hash") as executor:
            futures = {
                executor.submit(sha256_file, source_path): (tag, source_path)
                for tag, source_path in source_entries
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Hashing dataset images",
                unit="image",
            ):
                tag, source_path = futures[future]
                try:
                    hashes_by_source[(tag, source_path)] = future.result()
                except Exception as error:
                    logger.warning("hash_error | %s | %s: %s", source_path, type(error).__name__, error)

    pending_by_key: dict[tuple[str, str], AuditedImage] = {}
    seen_source_keys: set[tuple[str, str]] = set()
    for tag, source_path in source_entries:
        digest = hashes_by_source.get((tag, source_path))
        if digest is None:
            continue
        previous_tag = tag_by_digest.get(digest)
        if previous_tag is not None and previous_tag != tag:
            message = (
                f"Cross-tag duplicate SHA-256 conflict: {source_path} has the same content as "
                f"tag '{previous_tag}'"
            )
            logger.error(message)
            raise ProjectError(message)
        tag_by_digest[digest] = tag
        key = (tag, digest)
        if key in seen_source_keys:
            logger.warning("Skipping duplicate content in tag '%s': %s", tag, source_path)
            continue
        seen_source_keys.add(key)

        source_rel_path = source_path.relative_to(dataset_root).as_posix()
        storage_record = records_by_key.get(key)
        if storage_record is None:
            cache_path = cache_images_root / tag / f"{digest}.webp"
            storage_record = AuditedImage(
                source_path,
                source_rel_path,
                cache_path.relative_to(project_root).as_posix(),
                digest,
                tag,
            )
            records_by_key[key] = storage_record
            index_needs_write = True
        active_record = AuditedImage(
            source_path,
            source_rel_path,
            storage_record.cache_rel_path,
            digest,
            tag,
        )
        active_by_key[key] = active_record
        if not (project_root / storage_record.cache_rel_path).is_file():
            pending_by_key[key] = active_record

    created_cache_count = 0
    if pending_by_key:
        logger.info(
            "Building %d missing image caches with %d worker threads", len(pending_by_key), worker_count
        )

        def build_cache(record: AuditedImage) -> None:
            image = _load_rgb_checked(record.source_path, draft_long_side=cache_long_side)
            cache_path = project_root / record.cache_rel_path
            _cache_image(image, cache_path, cache_long_side)
            if not cache_path.is_file():
                raise ProjectError(f"Cache path is not a regular file: {cache_path}")

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cache-build") as executor:
            futures = {executor.submit(build_cache, record): (key, record) for key, record in pending_by_key.items()}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building image cache",
                unit="image",
            ):
                key, record = futures[future]
                try:
                    future.result()
                    created_cache_count += 1
                except OverflowError as error:
                    logger.warning("oversize | %s | %s", record.source_path, error)
                    active_by_key.pop(key, None)
                except Exception as error:
                    logger.warning("decode_error | %s | %s: %s", record.source_path, type(error).__name__, error)
                    active_by_key.pop(key, None)

    persisted_records = {
        key: record
        for key, record in records_by_key.items()
        if (project_root / record.cache_rel_path).is_file()
    }
    if len(persisted_records) != len(records_by_key):
        index_needs_write = True
    if index_needs_write:
        body = "".join(
            stable_json_dumps(
                {
                    "source_rel_path": record.source_rel_path,
                    "cache_rel_path": record.cache_rel_path,
                    "sha256": record.sha256,
                    "tag": record.tag,
                }
            )
            + "\n"
            for _, record in sorted(persisted_records.items(), key=lambda pair: pair[0])
        )
        atomic_write_text(cache_index_path, body)

    audited = sorted(active_by_key.values(), key=lambda item: (item.tag, item.sha256, item.source_rel_path))
    available_tags = sorted({item.tag for item in audited})
    if not audited:
        if not dataset_root.is_dir():
            raise ProjectError(
                f"Dataset directory does not exist and no usable cache index was found: {dataset_root}"
            )
        raise ProjectError("No decodable source image or usable cached image remained after auditing")
    missing_tags = sorted(declared_tags - set(available_tags))
    if missing_tags:
        raise ProjectError(f"Tags without a valid source image or usable cache: {missing_tags}")
    logger.info(
        "Cache ready: %d reusable images, %d newly built", len(audited) - created_cache_count, created_cache_count
    )

    if not create_split_manifest:
        return [], available_tags

    manifest_path = task_log_dir / "split_manifest.jsonl"
    existing = _read_manifest(manifest_path)
    additions = _assign_new_splits(audited, existing, split_config)
    if additions or not manifest_path.exists():
        _write_manifest(manifest_path, [*existing, *additions])
    assignments = {(item.tag, item.sha256): item.split for item in [*existing, *additions]}
    # Content identity, not filename, prevents a rename from crossing split boundaries.
    active = [
        ManifestItem(item.source_rel_path, item.cache_rel_path, item.sha256, item.tag,
                     assignments[(item.tag, item.sha256)])
        for item in audited
    ]
    logger.info(
        "Audit complete: %d valid images, %d tags, %d train, %d val, %d calibration, %d test",
        len(active),
        len(available_tags),
        sum(item.split == "train" for item in active),
        sum(item.split == "val" for item in active),
        sum(item.split == "calibration" for item in active),
        sum(item.split == "test" for item in active),
    )
    return active, available_tags



def _assign_new_splits(
    audited: Sequence[AuditedImage], existing: Sequence[ManifestItem], config: DataConfig,
) -> list[ManifestItem]:
    """Allocate only unseen content; never promote previously trained data to holdout.

    Validation receives the first scarce holdout slot, then test and calibration.
    Existing assignments (including temporarily unavailable content) remain immutable.
    ``stratified`` fills per-class target deficits using new images only. A class
    with at least three images retains its configured training share and at
    least two training examples where possible.
    """
    assignments: dict[tuple[str, str], str] = {}
    digest_tags: dict[str, str] = {}
    for item in existing:
        key = (item.tag, item.sha256)
        if key in assignments or (item.sha256 in digest_tags and digest_tags[item.sha256] != item.tag):
            raise ProjectError("Split manifest contains duplicate or conflicting content identities")
        assignments[key] = item.split
        digest_tags[item.sha256] = item.tag
    by_tag: dict[str, list[AuditedImage]] = defaultdict(list)
    for item in audited:
        if item.sha256 in digest_tags and digest_tags[item.sha256] != item.tag:
            raise ProjectError("Previously assigned content changed tags; review the split manifest")
        by_tag[item.tag].append(item)
    additions: list[ManifestItem] = []
    fractions = {"val": config.val_fraction, "calibration": config.calibration_fraction,
                 "test": config.test_fraction}
    for tag, candidates in sorted(by_tag.items()):
        counts = {split: 0 for split in ("train", *fractions)}
        unseen: list[AuditedImage] = []
        for item in sorted(candidates, key=lambda item: item.sha256):
            split = assignments.get((tag, item.sha256))
            if split is None:
                unseen.append(item)
            else:
                counts[split] += 1
        minimum_train = (
            max(2, math.ceil((1 - sum(fractions.values())) * len(candidates)))
            if len(candidates) >= 3 else 1
        )
        while counts["train"] < minimum_train and unseen:
            item = unseen.pop()
            additions.append(ManifestItem(item.source_rel_path, item.cache_rel_path,
                                          item.sha256, tag, "train"))
            counts["train"] += 1
        if counts["train"] == 0:
            raise ProjectError(f"Class '{tag}' has no remaining train image; add new training data")
        targets = {
            split: (max(1, math.floor(fraction * len(candidates))) if fraction > 0 else 0)
            for split, fraction in fractions.items()
        }
        for item in unseen:
            split = "train"
            if config.split_growth == "stratified" or not existing:
                split = next((name for name in ("val", "test", "calibration")
                              if counts[name] < targets[name]), "train")
            elif counts["val"] == 0 and config.val_fraction > 0:
                split = "val"  # Legacy minimum-one validation behaviour.
            counts[split] += 1
            additions.append(ManifestItem(item.source_rel_path, item.cache_rel_path,
                                          item.sha256, tag, split))
    return additions


@dataclass(frozen=True)
class SquareLetterbox:
    """Resize without distortion or cropping and pad to a square in RGB space."""

    size: int

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=tuple(round(value * 255) for value in IMAGENET_MEAN))


def build_transforms(
    input_size: int, *, config: DataConfig | None = None,
) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    """Build two-view training and deterministic evaluation preprocessing.

    Defaults reproduce the original augmentation. Aspect-ratio preservation uses
    letterboxing in both paths and must also be applied by registered inference.
    """
    config = config or DataConfig(input_size=input_size)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    strength = config.color_jitter_strength
    spatial = [SquareLetterbox(input_size)] if config.preserve_aspect_ratio else []
    train_transform = transforms.Compose(
        [
            *spatial,
            transforms.RandomResizedCrop(
                input_size, scale=(config.crop_min_scale, config.crop_max_scale),
                ratio=(1.0, 1.0) if config.preserve_aspect_ratio
                else (config.crop_min_ratio, config.crop_max_ratio),
            ),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    config.color_jitter_brightness * strength,
                    config.color_jitter_contrast * strength,
                    config.color_jitter_saturation * strength,
                    config.color_jitter_hue * strength,
                )],
                p=config.color_jitter_probability,
            ),
            transforms.RandomApply([transforms.RandomGrayscale(p=1.0)], p=config.grayscale_probability),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=config.erasing_probability,
                scale=(config.erasing_min_scale, config.erasing_max_scale),
            ),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquareLetterbox(input_size) if config.preserve_aspect_ratio
            else transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


class CachedImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]]):
    """Reads only prebuilt cache files; training returns two views per original."""

    def __init__(
        self,
        project_root: Path,
        items: Sequence[ManifestItem],
        tag_to_id: Mapping[str, int],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        two_views: bool,
        include_path: bool = False,
    ) -> None:
        self.project_root = project_root
        self.items = list(items)
        self.tag_to_id = dict(tag_to_id)
        self.transform = transform
        self.two_views = two_views
        self.include_path = include_path
        if not self.items:
            raise ProjectError("Dataset split has no images")
        missing_tags = {item.tag for item in self.items} - set(self.tag_to_id)
        if missing_tags:
            raise ProjectError(f"Missing tag ids for: {sorted(missing_tags)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        item = self.items[index]
        cache_path = self.project_root / item.cache_rel_path
        try:
            with Image.open(cache_path) as cached:
                image = cached.convert("RGB")
        except Exception as error:
            raise ProjectError(f"Cached image cannot be decoded: {cache_path}: {error}") from error
        label = torch.tensor(self.tag_to_id[item.tag], dtype=torch.long)
        first = self.transform(image)
        if self.two_views:
            return first, self.transform(image), label
        if self.include_path:
            return first, label, item.source_rel_path
        return first, label


class ClassBalancedSampler(Sampler[int]):
    """Draw classes, cycling through their shuffled examples before reuse.

    ``sqrt`` preserves the original distribution; ``uniform`` gives each class the
    same mass; ``effective_number`` applies per-example effective-number weights.
    """

    def __init__(
        self, items: Sequence[ManifestItem], tag_to_id: Mapping[str, int], seed: int,
        micro_batch: int, *, strategy: str = "sqrt", beta: float = 0.999,
    ) -> None:
        if not items or type(micro_batch) is not int or micro_batch < 1:
            raise ProjectError("Sampler requires nonempty items and a positive micro_batch")
        if strategy not in ("sqrt", "uniform", "effective_number"):
            raise ProjectError(f"Unsupported class sampling strategy: {strategy}")
        if type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta < 1:
            raise ProjectError("Effective-number beta must be finite and in [0, 1)")
        if any(item.tag not in tag_to_id for item in items):
            raise ProjectError("Sampler items contain an unmapped tag")
        self.indices_by_tag: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            self.indices_by_tag[item.tag].append(index)
        self.tags = sorted(self.indices_by_tag, key=lambda tag: tag_to_id[tag])
        counts = torch.tensor([len(self.indices_by_tag[tag]) for tag in self.tags], dtype=torch.float64)
        if strategy == "sqrt":
            weights = torch.sqrt(counts)
        elif strategy == "uniform":
            weights = torch.ones_like(counts)
        elif beta == 0:
            weights = counts
        else:
            # Class mass = n * per-example weight. expm1 avoids cancellation near beta=1.
            weights = counts * (1 - beta) / (-torch.expm1(counts * math.log(beta)))
        self.probabilities = weights / weights.sum()
        self.num_samples = max(len(items), micro_batch)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        class_ids = torch.multinomial(self.probabilities, self.num_samples, replacement=True, generator=generator)
        orders: dict[int, list[int]] = {}
        offsets: dict[int, int] = {}
        for class_id in class_ids.tolist():
            samples = self.indices_by_tag[self.tags[class_id]]
            if class_id not in orders or offsets[class_id] == len(samples):
                orders[class_id] = torch.randperm(len(samples), generator=generator).tolist()
                offsets[class_id] = 0
            yield samples[orders[class_id][offsets[class_id]]]
            offsets[class_id] += 1

    def __len__(self) -> int:
        return self.num_samples


class SqrtClassSampler(ClassBalancedSampler):
    """Backward-compatible sampler retaining the original sqrt defaults."""


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    sampler: Sampler[int] | None = None,
    shuffle: bool | None = None,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 1,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": sampler is None if shuffle is None else shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        # Keep reusable evaluation workers alive between epochs; training loaders are
        # rebuilt each epoch for deterministic epoch-specific augmentation, so workers
        # receive the usual epoch-specific seed and shut down with their loader.
        "persistent_workers": num_workers > 0 and persistent_workers,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def source_paths(items: Iterable[ManifestItem], project_root: Path, dataset: str) -> list[Path]:
    dataset_root = project_root / "datasets" / dataset
    return [dataset_root / item.source_rel_path for item in items]
