# Image Auto Classifier

[English](README.md) | [简体中文](README.zh-CN.md)

Image Auto Classifier is a local image classification and dataset-maintenance workflow built around **DINOv3 ViT-B/16 + LoRA**.

The project trains an independent classifier for each dataset, registers the best model for inference, rejects low-confidence or low-similarity predictions, exports uncertain samples for active learning, and provides a Tkinter GUI for manually reviewing, correcting, deleting, and archiving classification results.

The full workflow is designed around local directories:

```text
datasets → cache → training → registered models
                              ↓
input → inference → output → manual review → datasets
                         ↘ active learning
```

## Features

* DINOv3 ViT-B/16 image encoder with a frozen backbone and targeted LoRA adaptation.
* LoRA applied to the attention and MLP projections in transformer blocks 9–12.
* `768 → 512 → 256` normalized metric embedding head.
* ArcFace classification with supervised contrastive learning.
* EMA-based evaluation and inference weights.
* Deterministic training and resumable epoch checkpoints.
* Durable image cache with SHA-256 content tracking.
* Stable train/validation split manifests across training runs.
* Automatic adaptation when new dataset content or classes are added.
* Temperature calibration and automatically selected probability thresholds.
* Per-class prototype similarity gates for rejection of uncertain predictions.
* Single-dataset and multi-model unified inference.
* Active-learning CSV export for rejected samples.
* Atomic best-model registration for safe concurrent inference.
* Tkinter-based manual review and dataset organization workflow.
* HEIC, JPEG, PNG, and WebP support for training and inference.

## Tech Stack

| Component                  | Technology                                 |
| -------------------------- | ------------------------------------------ |
| Language                   | Python 3.10+                               |
| Deep learning              | PyTorch, torchvision                       |
| Backbone                   | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Model integration          | Hugging Face Transformers                  |
| Parameter-efficient tuning | PEFT / LoRA                                |
| Image processing           | Pillow, pillow-heif                        |
| Metrics                    | scikit-learn                               |
| Configuration              | PyYAML                                     |
| Training progress          | tqdm                                       |
| Training plots             | Matplotlib                                 |
| Manual review UI           | Tkinter                                    |
| Numerical operations       | NumPy                                      |

The GUI also supports `send2trash` for recycle-bin deletion when installed and `pypinyin` for grouping non-Latin label names by initial letter when available.

## Architecture

### Training pipeline

Training starts from a YAML task configuration:

```text
config
  │
  ├─ dataset audit
  │    ├─ validate images
  │    ├─ SHA-256 deduplication
  │    ├─ lossless WebP cache
  │    └─ persistent split manifest
  │
  ├─ DINOv3 ViT-B/16
  │    └─ LoRA on blocks 9–12
  │
  ├─ 768 → 512 → 256 metric head
  │
  ├─ ArcFace + supervised contrastive loss
  │
  ├─ EMA evaluation
  │
  ├─ calibration + class prototypes
  │
  └─ checkpoint + registered best model
```

The DINOv3 backbone remains frozen. Only LoRA parameters, the metric head, and ArcFace classifier are trained.

Training uses two augmented views of every sampled image and optimizes:

```text
0.75 × ArcFace cross-entropy
+ 0.25 × supervised contrastive loss
```

LoRA parameters are enabled after the first three schedule epochs. The optimizer uses separate learning rates for LoRA and head parameters and applies a three-epoch linear warm-up followed by cosine decay.

### Inference pipeline

For each input image, inference computes:

1. the predicted class,
2. calibrated maximum class probability,
3. cosine similarity to the predicted class prototype.

A result is accepted only when both the probability threshold and the class-specific similarity threshold are satisfied.

Accepted images are copied to:

```text
output/<dataset>/<tag>/
```

Rejected images can be ranked for manual labeling through the active-learning export.

### Model registry

The active model for a dataset is resolved through:

```text
models/<dataset>/
├── current.json
└── versions/
    └── <registry-version>/
        ├── best.ckpt
        └── metadata.json
```

A new best model is published into an immutable version directory first. `current.json` is then atomically switched to the new version, preventing inference from observing a partially written model.

Stale registry versions are removed after publication, so the registry retains the currently active best model rather than historical rollback versions.

## Project Structure

```text
.
├── configs/
│   └── task.yaml
├── scripts/
│   ├── train.py
│   └── inference.py
├── src/
│   └── image_auto_classifier/
│       ├── __init__.py
│       ├── checkpointing.py
│       ├── common.py
│       ├── config.py
│       ├── data.py
│       ├── inferencer.py
│       ├── losses.py
│       ├── metrics.py
│       ├── model.py
│       └── trainer.py
└── manual_classify_gui.py
```

Key modules:

| Module                   | Responsibility                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `config.py`              | YAML parsing, defaults, and strict configuration validation                        |
| `data.py`                | Dataset auditing, cache generation, split manifests, transforms, sampling, loaders |
| `model.py`               | DINOv3 loading, exact LoRA targeting, metric head, ArcFace model, EMA              |
| `losses.py`              | ArcFace layer and supervised contrastive objective                                 |
| `metrics.py`             | Evaluation, calibration, prototypes, similarity gates, active-learning selection   |
| `trainer.py`             | End-to-end training, resume/adaptation logic, evaluation, early stopping           |
| `checkpointing.py`       | Checkpoint schema, atomic persistence, best-model registry                         |
| `inferencer.py`          | Registered-model loading, inference, rejection gates, result copying               |
| `manual_classify_gui.py` | Manual result review, correction, deletion, and archive workflow                   |

## Prerequisites

### Python

The source uses Python syntax and standard-library features that require **Python 3.10 or newer**.

### NVIDIA CUDA GPU

Training and inference explicitly require CUDA:

```text
CUDA GPU is required
```

CPU-only execution is not supported by the current training or inference entry points.

Mixed-precision CUDA training is part of the fixed training configuration.

### DINOv3 access

The project uses:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

This Hugging Face repository is access-gated.

Before the first training run:

1. request/accept access to the model on Hugging Face;
2. authenticate the machine using:

```bash
hf auth login
```

Alternatively, provide a Hugging Face token through:

```bash
HF_TOKEN=<your-token>
```

The standard Hugging Face endpoint is used by default. `HF_ENDPOINT` may be set when an alternative endpoint is required.

Training must download/populate the DINOv3 files at least once. Registered-model inference reconstructs the model using the local Hugging Face cache.

## Dataset Layout

Each training dataset lives under:

```text
datasets/<dataset>/
```

Every first-level directory is treated as one classification tag:

```text
datasets/
└── task1/
    ├── character_a/
    │   ├── 001.jpg
    │   └── 002.webp
    ├── character_b/
    │   ├── 003.png
    │   └── 004.heic
    └── character_c/
        └── 005.jpeg
```

Training images must be directly contained in their tag directory.

Supported source extensions are:

```text
.png
.jpg
.jpeg
.heic
.webp
```

During auditing, images are decoded, EXIF orientation is applied, and valid samples are stored as lossless WebP cache files.

The cache layout is:

```text
cache/<dataset>/
├── cache_index.jsonl
└── images/
    └── <tag>/
        └── <sha256>.webp
```

The cache index preserves tag and content metadata, allowing cached data to remain usable independently of the original source directory.

Identical content inside the same tag is deduplicated. Identical content appearing under different tags is treated as a labeling conflict and stops the audit.

## Configuration

A training task is defined by a YAML file in `configs/`.

For example:

```yaml
dataset: task1
seed: 42

data:
  cache_long_side: 1280
  input_size: 256
  num_workers: 2

train:
  max_epochs: 50
  micro_batch: 8
  grad_accum: 4
  mixed_precision: true
  early_stop_patience: 10

model:
  name: facebook/dinov3-vitb16-pretrain-lvd1689m
  lora_rank: 16
  lora_alpha: 32
```

The configuration filename determines the task name. For example:

```text
configs/task1.yaml
```

creates the task:

```text
task1
```

while its `dataset` field selects:

```text
datasets/task1/
```

### Configuration reference

| Key                         |         Default | Description                                           |
| --------------------------- | --------------: | ----------------------------------------------------- |
| `dataset`                   |        required | Dataset directory and model-registry name             |
| `seed`                      |          `3407` | Global deterministic seed                             |
| `data.cache_long_side`      |          `1280` | Maximum long side of cached images                    |
| `data.input_size`           |           `256` | Model input resolution                                |
| `data.num_workers`          |             `2` | DataLoader worker count                               |
| `train.max_epochs`          |            `50` | Maximum normal training epochs                        |
| `train.micro_batch`         |             `8` | Training micro-batch size                             |
| `train.grad_accum`          |             `4` | Gradient accumulation steps                           |
| `train.mixed_precision`     |          `true` | CUDA AMP; fixed to `true`                             |
| `train.early_stop_patience` |            `10` | Validation non-improvement patience                   |
| `train.adaptation_epochs`   |            `12` | Maximum adaptation cycle length after dataset changes |
| `model.name`                | DINOv3 ViT-B/16 | Fixed backbone                                        |
| `model.lora_rank`           |            `16` | Fixed LoRA rank                                       |
| `model.lora_alpha`          |            `32` | Fixed LoRA alpha                                      |

The model name, LoRA rank, LoRA alpha, and mixed-precision mode are intentionally locked. Invalid overrides are rejected during configuration parsing.

## Training

Run commands from the repository root.

Train a task:

```bash
python scripts/train.py --config configs/task1.yaml
```

Other included task configurations can be started in the same way:

```bash
python scripts/train.py --config configs/task1.yaml
```

### Resume training

Resume from the latest epoch checkpoint belonging to the same task:

```bash
python scripts/train.py \
  --config configs/task1.yaml \
  --resume auto
```

Resume validates the saved dataset, preprocessing, model, LoRA, optimizer, scheduler, scaler, and RNG state before continuing.

Removing an existing tag while resuming is rejected because the saved classifier and prototype state would no longer be valid.

When the dataset changes by adding images or classes, training enters a bounded adaptation cycle. Existing class IDs are preserved and newly added classes are appended.

### Training artifacts

A task such as `task1` writes artifacts under:

```text
logs/task1/
├── task.log
├── curves.png
├── split_manifest.jsonl
└── models/
    ├── best.ckpt
    └── epoch_XXX.ckpt
```

Only the latest three epoch checkpoints are retained.

The split manifest is persistent. Existing samples keep their previous train/validation assignment when more images are added.

When validation data exists, model selection uses macro F1. Classes reaching at least 95% validation accuracy are excluded from subsequent training sampling, and training can stop once every class reaches that threshold.

## Inference

Inference reads images directly from the first level of the requested input directory.

The default input directory is:

```text
input/
```

### Use one registered dataset model

```bash
python scripts/inference.py \
  --dataset task1 \
  --input input
```

Accepted images are copied to:

```text
output/task1/<predicted-tag>/
```

### Unified inference across all registered models

If `--dataset` is omitted, inference uses unified mode:

```bash
python scripts/inference.py --input input
```

This is equivalent to:

```bash
python scripts/inference.py \
  --dataset __UNITEINFER__ \
  --input input
```

Every registered dataset model is evaluated one at a time. For each image, the class with the highest maximum probability across the models becomes the candidate prediction. The winning model's probability and prototype-similarity gates still determine whether the image is accepted.

Models are loaded and released sequentially to avoid keeping multiple DINOv3 instances in GPU memory.

### Override the probability threshold

The registered model normally uses its calibrated probability threshold.

To override it:

```bash
python scripts/inference.py \
  --dataset task1 \
  --input input \
  --threshold 0.95
```

The override must be between `0` and `1`.

The class-specific cosine-similarity gate remains active.

### Active learning

Rejected samples can be exported to:

```text
logs/<task>/active_learning/<timestamp>.csv
```

Each CSV contains fields including:

```text
rank
u
pmax
similarity
suggested_tag
path
```

The selector first prioritizes uncertainty and then uses a greedy k-center strategy against labeled embeddings to retain a diverse set of at most 200 samples.

## Manual Review GUI

Start the review application from the repository root:

```bash
python manual_classify_gui.py
```

The GUI operates primarily on:

```text
output/<dataset>/<tag>/
```

and provides:

* dataset and predicted-tag navigation;
* responsive thumbnail browsing;
* multi-selection and inverse selection;
* full-size image preview;
* independent destination dataset selection;
* destination tag selection with alphabetical navigation;
* creation of new destination folders;
* moving images between output tags/datasets;
* removal from `output` only;
* deletion from `output` together with matching input files;
* optional recycle-bin deletion through `send2trash`;
* recursive archive of reviewed output back into `datasets`.

Useful shortcuts:

```text
Ctrl+Tab        next output tag directory
Ctrl+Shift+Tab  previous output tag directory
```

### Review and archive workflow

A typical workflow is:

```text
1. Run inference
2. Open manual_classify_gui.py
3. Review output/<dataset>/<tag>
4. Move incorrectly classified images to the desired dataset/tag
5. Remove or delete unwanted images
6. Archive reviewed output
```

`Archive all output` recursively merges:

```text
output/<dataset>/
```

into:

```text
datasets/<dataset>/
```

Filename collisions are preserved by assigning a unique destination name instead of overwriting an existing file.

The GUI cleans matching source files from `input` only after all output files involved in the archive have been moved successfully.

## Runtime Directories

The project creates and consumes the following runtime directories:

```text
datasets/   labeled training data
cache/      durable decoded image cache
input/      images awaiting inference
output/     accepted inference results and manual-review workspace
logs/       task logs, manifests, plots, checkpoints, active-learning CSVs
models/     registered best models used by inference
```

These directories are separate from the Python source tree and are created as needed by the training, inference, and review workflows.

## License

This project is licensed under the MIT License.
