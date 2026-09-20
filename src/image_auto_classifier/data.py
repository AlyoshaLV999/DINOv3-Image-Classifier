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
        if item.split not in {"train", "val"}:
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

    manifest_path = task_log_dir / "split_manifest.jsonl"
    existing = _read_manifest(manifest_path)
    existing_keys = {(item.source_rel_path, item.sha256): item for item in existing}
    active_keys = {(item.source_rel_path, item.sha256) for item in audited}
    additions: list[ManifestItem] = []
    existing_active_by_tag: dict[str, list[ManifestItem]] = defaultdict(list)
    new_by_tag: dict[str, list[AuditedImage]] = defaultdict(list)
    for item in existing:
        if (item.source_rel_path, item.sha256) in active_keys:
            existing_active_by_tag[item.tag].append(item)
    for item in audited:
        if (item.source_rel_path, item.sha256) not in existing_keys:
            new_by_tag[item.tag].append(item)

    if not existing:
        for tag in available_tags:
            candidates = sorted(
                (item for item in audited if item.tag == tag), key=lambda item: item.sha256
            )
            val_count = 0 if len(candidates) == 1 else max(1, math.floor(0.2 * len(candidates)))
            for position, item in enumerate(candidates):
                additions.append(
                    ManifestItem(
                        item.source_rel_path,
                        item.cache_rel_path,
                        item.sha256,
                        item.tag,
                        "val" if position < val_count else "train",
                    )
                )
    else:
        for tag in sorted(new_by_tag):
            candidates = new_by_tag[tag]
            old_items = existing_active_by_tag.get(tag, [])
            old_train_count = sum(item.split == "train" for item in old_items)
            old_val_count = sum(item.split == "val" for item in old_items)
            for position, item in enumerate(sorted(candidates, key=lambda candidate: candidate.sha256)):
                split = "val" if old_train_count == 1 and old_val_count == 0 and position == 0 else "train"
                additions.append(
                    ManifestItem(item.source_rel_path, item.cache_rel_path, item.sha256, item.tag, split)
                )

    if additions:
        _write_manifest(manifest_path, [*existing, *additions])
    elif not manifest_path.exists():
        _write_manifest(manifest_path, existing)

    final_items = _read_manifest(manifest_path)
    active = [item for item in final_items if (item.source_rel_path, item.sha256) in active_keys]
    if len(active) != len(audited):
        raise ProjectError("Split manifest does not cover every audited image")
    logger.info(
        "Audit complete: %d valid images, %d tags, %d train, %d val",
        len(active),
        len(available_tags),
        sum(item.split == "train" for item in active),
        sum(item.split == "val" for item in active),
    )
    return active, available_tags



def build_transforms(input_size: int) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.70, 1.00), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.20, 0.20, 0.20, 0.08)], p=0.8),
            transforms.RandomApply([transforms.RandomGrayscale(p=1.0)], p=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), antialias=True),
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


class SqrtClassSampler(Sampler[int]):
    """Replacement sampler with class probability proportional to sqrt(train count)."""

    def __init__(self, items: Sequence[ManifestItem], tag_to_id: Mapping[str, int], seed: int, micro_batch: int) -> None:
        self.indices_by_tag: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            self.indices_by_tag[item.tag].append(index)
        self.tags = sorted(self.indices_by_tag, key=lambda tag: tag_to_id[tag])
        counts = torch.tensor([len(self.indices_by_tag[tag]) for tag in self.tags], dtype=torch.float64)
        self.probabilities = torch.sqrt(counts) / torch.sqrt(counts).sum()
        self.num_samples = max(len(items), micro_batch)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        class_ids = torch.multinomial(self.probabilities, self.num_samples, replacement=True, generator=generator)
        for class_id in class_ids.tolist():
            samples = self.indices_by_tag[self.tags[class_id]]
            position = torch.randint(len(samples), (1,), generator=generator).item()
            yield samples[position]

    def __len__(self) -> int:
        return self.num_samples


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
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": sampler is None if shuffle is None else shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def source_paths(items: Iterable[ManifestItem], project_root: Path, dataset: str) -> list[Path]:
    dataset_root = project_root / "datasets" / dataset
    return [dataset_root / item.source_rel_path for item in items]


