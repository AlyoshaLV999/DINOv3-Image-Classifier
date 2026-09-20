# DINOv3 Image Classifier

[中文说明见下方](#中文说明)

An end-to-end, GPU-accelerated image classification workflow built on **DINOv3**, **LoRA**, and **ArcFace**. It covers dataset auditing, reproducible training, atomic model registration, confidence-aware batch inference, active-learning candidate export, and a desktop interface for human review.

This repository is a portfolio-ready reference implementation for a practical human-in-the-loop image classification pipeline. It does not include datasets, trained checkpoints, or model credentials.

## Features

- **Dataset audit and caching**: validates images, corrects EXIF orientation, detects SHA-256 duplicates, and creates reusable WebP caches.
- **Reproducible training**: deterministic seeding, immutable train/validation manifests, mixed precision, gradient accumulation, EMA evaluation, early stopping, and resumable checkpoints.
- **Efficient adaptation**: freezes the DINOv3 backbone and applies LoRA to selected transformer blocks.
- **Metric-aware classification**: combines a 256-dimensional embedding head, ArcFace classification, supervised contrastive loss, and class prototypes.
- **Safe model registry**: atomically publishes a complete checkpoint and metadata pair before switching the active model pointer.
- **Selective inference**: accepts predictions only when both probability and class-level cosine-similarity gates pass; uncertain examples can be exported for active learning.
- **Human review GUI**: a Tkinter workspace for browsing, moving, deleting, and archiving predicted images.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Deep learning | PyTorch, torchvision, CUDA AMP |
| Foundation model | Hugging Face Transformers, DINOv3 ViT-B/16 |
| Fine-tuning | PEFT / LoRA |
| Metrics | ArcFace, supervised contrastive loss, scikit-learn |
| Image processing | Pillow, pillow-heif |
| Configuration & tooling | PyYAML, tqdm, matplotlib, Tkinter |

## Project Structure

```text
.
├── configs/                     # Training configurations
├── docs/images/                 # Add screenshots or GIF demos here
├── scripts/
│   ├── train.py                 # Training CLI
│   └── inference.py             # Batch inference CLI
├── src/image_auto_classifier/
│   ├── data.py                  # Audit, cache, split, DataLoader
│   ├── model.py                 # DINOv3, LoRA, ArcFace, EMA
│   ├── trainer.py               # Training and model publishing
│   ├── inferencer.py            # Registry-backed inference
│   └── ...
├── manual_classify_gui.py       # Human-review desktop application
├── requirements.txt
└── .env.example
```

Runtime artifacts are intentionally excluded from version control:

```text
datasets/<dataset>/<label>/      # Local labelled images
input/                           # Images waiting for inference
cache/                           # Derived image cache
logs/                            # Logs, curves, and task checkpoints
models/                          # Registered model weights and metadata
output/                          # Predicted images awaiting review
```

## Installation

### Prerequisites

- Python 3.10 or later
- An NVIDIA GPU with a CUDA-compatible PyTorch installation
- Access to the gated Hugging Face model `facebook/dinov3-vitb16-pretrain-lvd1689m`

> Install the PyTorch build appropriate for your CUDA environment first. See the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

```bash
git clone https://github.com/<your-github-username>/dinov3-image-classifier.git
cd dinov3-image-classifier

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

After accepting the model's access terms, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Alternatively, copy `.env.example` to `.env` and set `HF_TOKEN` in your shell or environment manager. The application relies on the standard Hugging Face authentication flow and does not read `.env` automatically.

## Usage

### 1. Prepare a dataset

Place labelled images in one directory per class:

```text
datasets/
└── sample-images/
    ├── class-a/
    │   ├── image_001.jpg
    │   └── image_002.png
    └── class-b/
        └── image_003.webp
```

Use `configs/example.yaml` as a starting point. The dataset field must match the directory name under `datasets/`.

### 2. Train and register a model

```bash
python scripts/train.py --config configs/example.yaml
python scripts/train.py --config configs/example.yaml --resume auto
```

The best model is published under `models/<dataset>/` through an atomic registry pointer.

### 3. Run batch inference

Put images directly in `input/`, then run one registered dataset model:

```bash
python scripts/inference.py --dataset sample-images --input input
```

Or compare every registered dataset model:

```bash
python scripts/inference.py --dataset __UNITEINFER__ --input input
```

Accepted images are copied to `output/<dataset>/<predicted-label>/`. Rejected or uncertain samples can produce active-learning CSV files in `logs/<task>/active_learning/`.

### 4. Review predictions manually

```bash
python manual_classify_gui.py
```

Run the GUI from the repository root. It lets you inspect result thumbnails, move images into target label directories, remove or delete selections, and archive reviewed results back into `datasets/`.

## Demo Assets

Add anonymized screenshots or a short GIF to `docs/images/`, then reference them here:

```md
![Training curves](docs/images/training-curves.png)
![Review workspace](docs/images/review-workspace.gif)
```

## Portfolio Highlights

- Separates training data, derived caches, model registry, inference input, and review output to make the workflow auditable.
- Uses strict configuration validation and checkpoint schema checks to prevent accidental incompatible resumes.
- Combines confidence calibration with per-class similarity thresholds to avoid forcing low-quality images into a label.
- Builds a complete data feedback loop: inference → uncertainty queue → human review → dataset archive → retraining.
- Uses atomic writes for checkpoints, registry metadata, and prediction reports to reduce partial-write failure modes.

## Roadmap

- [ ] Add automated unit and integration tests.
- [ ] Add sample images and an anonymized demo configuration.
- [ ] Add CI for formatting, linting, and test execution.
- [ ] Add benchmark results and hardware/memory profiles.
- [ ] Add a web-based review interface as an optional frontend.

## License

Released under the [MIT License](LICENSE). Before publishing, replace the copyright holder placeholder in `LICENSE` with your legal name or organization.

---

# 中文说明

`DINOv3 Image Classifier` 是一个基于 **DINOv3、LoRA 与 ArcFace** 的端到端图像分类工作流，面向 GPU 环境下的训练、模型注册、批量推理、低置信度样本筛选和人工复核整理。仓库不包含数据集、训练权重或任何访问凭据。

## 核心功能

- 对训练数据进行图片校验、EXIF 方向修正、SHA-256 去重和 WebP 缓存。
- 使用固定数据划分、AMP、梯度累积、EMA、早停和断点恢复，保证训练过程可追溯。
- 冻结 DINOv3 主干，并通过 LoRA 做参数高效微调；分类头结合 ArcFace 与监督式对比学习。
- 以原子方式发布模型权重与元数据，推理只加载已完整注册的模型版本。
- 以“分类概率 + 类别余弦相似度”双阈值拒识，降低错误归类风险，并导出主动学习候选样本。
- 提供 Tkinter 人工复核界面，可浏览、移动、删除和归档推理结果。

## 技术栈

PyTorch、CUDA AMP、torchvision、Hugging Face Transformers、DINOv3、PEFT/LoRA、Pillow、pillow-heif、scikit-learn、PyYAML、matplotlib、tqdm 和 Tkinter。

## 安装与运行

1. 准备 Python 3.10+、NVIDIA GPU 和与 CUDA 匹配的 PyTorch。
2. 在 Hugging Face 接受 `facebook/dinov3-vitb16-pretrain-lvd1689m` 的访问条款后执行 `huggingface-cli login`。
3. 创建虚拟环境并执行 `pip install -r requirements.txt`。
4. 将训练图片按 `datasets/<dataset>/<label>/` 组织，复制并修改 `configs/example.yaml`。
5. 训练：`python scripts/train.py --config configs/example.yaml`。
6. 推理：`python scripts/inference.py --dataset sample-images --input input`。
7. 人工复核：`python manual_classify_gui.py`。

项目结构、运行时目录、演示素材预留位置、项目亮点和后续计划均见英文部分；英文说明位于前半部分，便于 GitHub 访客与招聘者快速浏览。

## 许可证

本项目采用 [MIT License](LICENSE)。公开发布前，请将 `LICENSE` 中的版权占位符替换为你的真实姓名或组织名称。
