# DINOv3-Image-Classifier


[简体中文](#简体中文) | [English](#english)

---

## 简体中文

一个基于 **DINOv3 + LoRA** 的图像自动分类工具集，提供从数据审计、缓存、训练、推理到人工复核的端到端工作流。

### 简介

本项目以 Hugging Face 的 `facebook/dinov3-vitb16-pretrain-lvd1689m` 作为冻结主干，通过 LoRA 与自定义的 256 维度量头（ArcFace + 监督对比学习）在多标签数据集上训练分类器。训练结束后，模型会通过基于余弦相似度的原型拒绝门（prototype rejection gate）和温度校准的概率阈值对推理结果进行双重过滤，从而将输入图片自动归入 `output/<dataset>/<tag>`。

### 核心特性

- **DINOv3 + LoRA 微调**：冻结 ViT-B/16 主干，仅训练第 9–12 个 Transformer 块的 `qkv/proj/fc1/fc2` 投影（共 24 个物理目标），LoRA rank 16 / alpha 32 为固定方案。
- **度量学习头**：`768 → 512 → 256` L2 归一化嵌入，配合 ArcFace 与监督对比损失进行联合优化。
- **EMA 评估**：验证阶段使用指数移动平均参数，指标更稳定。
- **可扩展分类头**：新增类别时保留原有 ArcFace 权重与优化器状态，仅在需要时扩容。
- **类别级自适应训练门**：验证集上类别准确率 ≥ 95% 时该类别自动退出训练，全部达标则提前结束。
- **严格检查点与注册表**：模型以不可变版本发布至 `models/<dataset>/versions/`，通过原子 `current.json` 指针切换，推理端始终只看到完整版本；历史版本会自动清理。
- **可复用图片缓存**：源图片经哈希、去重、EXIF 修正后以无损 WebP 缓存，`cache/<dataset>/cache_index.jsonl` 记录标签，任务间可复用。
- **不变的分割清单**：`split_manifest.jsonl` 一旦生成即冻结，避免训练/验证泄漏；数据集变化时进入有限 epochs 的适配周期。
- **拒绝门 + 主动学习**：推理时按概率阈值与每类相似度门双过滤，被拒绝的图片会写入 `logs/<task>/active_learning/*.csv`（基于不确定度预筛 + 贪心 k-center 选点）。
- **统一推理（`__UNITEINFER__`）**：跨所有已注册数据集模型比对最高概率，选出最佳归属。
- **低置信度旁路**：预测概率低于 0.20 的图片会被复制到 `output/OUTOFTAG/OUTOFTAG`（最多 300 张）供人工检视。
- **PySide6 人工复核 GUI**：支持跨数据集归类、缩略图缓存、Shift 连选、预览、移除、删除与一键归档。

### 技术栈

| 分类 | 组件 |
| --- | --- |
| 语言 | Python 3.10+（使用了 `X \| Y` 类型注解） |
| 深度学习 | PyTorch、torchvision |
| 主干网络 | Hugging Face Transformers (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| 微调 | PEFT (LoRA) |
| 图像处理 | Pillow、pillow-heif |
| 指标 | scikit-learn |
| 曲线绘制 | Matplotlib（Agg 后端） |
| GUI | PySide6，可选依赖 send2trash、pypinyin |
| 配置 | PyYAML |
| 其他 | NumPy、tqdm |

> 训练与推理严格要求 CUDA GPU（`common.require_cuda`），无 CPU 回退路径。

### 项目结构

```text
.
├── configs/                       # 每个任务一个 YAML 配置文件
│   ├── Tband.yaml
│   ├── Tgi1.yaml
│   ├── Thi3.yaml
│   ├── Thsr1.yaml
│   ├── Tpixiv1.yaml
│   └── Tzzz1.yaml
├── scripts/
│   ├── train.py                   # 训练入口
│   └── inference.py               # 推理入口
├── src/
│   └── image_auto_classifier
/
│       ├── __init__.py
│       ├── common.py              # 工具：文件原子写、RNG、日志、CUDA 检查
│       ├── config.py              # YAML 解析与严格校验
│       ├── data.py                # 数据集审计、缓存、分割、采样、DataLoader
│       ├── model.py               # DINOv3 + LoRA + ArcFace 组合模型，含 EMA
│       ├── losses.py              # ArcFace 与 SupCon 目标
│       ├── metrics.py             # 评估、温度校准、原型、主动学习选点
│       ├── checkpointing.py       # 检查点 schema、注册表、版本清理
│       ├── trainer.py             # 训练循环、调度器、早停、适配周期
│       └── inferencer.py          # 加载注册模型、批量推理、拒绝门、导出
└── manual_classify_gui.py         # PySide6 人工复核工具
```

运行时（首次运行自动创建）：

```text
datasets/<dataset>/<tag>/          # 原始图片
cache/<dataset>/images/<tag>/      # 无损 WebP 缓存 + cache_index.jsonl
models/<dataset>/                  # 注册表：current.json + versions/<id>/{best.ckpt, metadata.json}
logs/<task_name>/                  # 日志、split_manifest.jsonl、curves.png、models/*.ckpt、active_learning/*.csv
output/<dataset>/<tag>/            # 推理输出（自动创建）
output/OUTOFTAG/OUTOFTAG/          # 低置信度旁路
input/                             # 推理输入目录
```

### 快速开始

#### 1. 环境要求

- Python 3.10 或更高版本
- 一块可用的 NVIDIA GPU（CUDA 版 PyTorch）
- 能够访问 `facebook/dinov3-vitb16-pretrain-lvd1689m`：该仓库为 gated 模型，需要在 Hugging Face 上同意使用条款后通过 `hf auth login` 或设置 `HF_TOKEN` 完成认证

#### 2. 安装依赖

```bash
git clone <repository-url>
cd image_auto_classifier


# 建议先按 https://pytorch.org/ 的说明安装与 CUDA 匹配的 torch / torchvision
pip install torch torchvision
pip install transformers peft pillow pillow-heif pyyaml numpy scikit-learn matplotlib tqdm

# 如需使用人工复核 GUI
pip install PySide6
pip install send2trash pypinyin   # 可选：回收站支持与中文首字母分组
```

项目使用 `src/` 布局，`scripts/*.py` 与 `manual_classify_gui.py` 已自动将 `src/` 加入 `sys.path`，无需额外安装即可从仓库根目录运行。

#### 3. 准备数据

将源图片按标签分目录放置：

```text
datasets/<dataset>/
├── tag_a/
│   ├── 001.jpg
│   └── 002.png
└── tag_b/
    └── 003.webp
```

支持的图片后缀：`.png`、`.jpg`、`.jpeg`、`.heic`、`.webp`（训练审计）；GUI 额外支持 `.bmp`、`.gif`、`.tif`、`.tiff`。

首次执行训练时会自动完成：哈希计算、跨标签重复检测、EXIF 修正、无损 WebP 缓存、类别均衡的 80/20 划分，并写入 `logs/<task>/split_manifest.jsonl`。

#### 4. 配置任务

在 `configs/` 下新建 YAML，文件名（去掉扩展名）即任务名：

```yaml
dataset: band            # 对应 datasets/band，也是 output/ 与 models/ 下的子目录名
seed: 42
data:
  cache_long_side: 1280  # 缓存最长边，超出会被等比缩小
  input_size: 256        # 模型输入尺寸
  num_workers: 2
train:
  max_epochs: 50
  micro_batch: 8
  grad_accum: 4
  mixed_precision: true  # 固定为 true
  early_stop_patience: 10
  adaptation_epochs: 12  # 数据集变更后适配周期长度（可选，默认 12）
model:
  name: facebook/dinov3-vitb16-pretrain-lvd1689m
  lora_rank: 16          # 固定为 16
  lora_alpha: 32         # 固定为 32
```

`model.name`、`model.lora_rank`、`model.lora_alpha` 与 `train.mixed_precision` 已被锁定，改动会直接触发 `ConfigurationError`。`configs/` 目录中已提供 6 个示例配置。

#### 5. 训练

```bash
python scripts/train.py --config configs/Tband.yaml

# 从本任务最新 epoch 检查点续训
python scripts/train.py --config configs/Tband.yaml --resume auto
```

训练关键行为：

- 学习率策略：前 3 个 epoch 线性预热，随后余弦衰减至 `1e-6`。
- LoRA 在前 3 个 epoch 冻结，仅训练度量头与 ArcFace；第 4 个 epoch 起开放 LoRA。
- 每 epoch 使用 EMA 参数在验证集上计算 macro-F1，若连续 `early_stop_patience` 个 epoch 无 ≥ 0.002 的提升则提前停止。
- 每个 epoch 结束后落盘 `logs/<task>/models/epoch_NNN.ckpt`（只保留最近 3 个），验证指标改进时将当轮检查点注册为 `models/<dataset>/versions/<id>/`，并原子更新 `current.json`。
- `split_manifest.jsonl` 的哈希发生变化时，会自动从 `logs/<task>/models/best.ckpt` 出发执行 `adaptation_epochs` 个 epoch 的有限适配；新增标签会扩展 ArcFace，已删除标签会直接报错拒绝恢复。
- 崩溃恢复友好：优化器、调度器、GradScaler、RNG 状态均会随检查点保存与还原。

#### 6. 推理

```bash
# 使用全部已注册的数据集模型进行统一推理
python scripts/inference.py --dataset __UNITEINFER__ --input input

# 只使用某一个数据集模型
python scripts/inference.py --dataset band --input input

# 覆盖默认概率阈值（默认取注册时的校准阈值）
python scripts/inference.py --dataset band --input input --threshold 0.9
```

推理流程会：

1. 逐图进行 EMA 前向，得到 256 维嵌入与 logits。
2. 按类别选取对应原型，计算余弦相似度，与类别级门限比较。
3. 同时使用温度校准后的 softmax 概率与概率阈值进行过滤。
4. 通过两张门限的图片复制到 `output/<dataset>/<tag>/`；被拒绝的写入 `logs/<task>/active_learning/<timestamp>.csv`（最多 200 条）。
5. 概率低于 `0.20` 的图片额外复制到 `output/OUTOFTAG/OUTOFTAG/`（最多 300 张）。
6. 输出 `accepted=N rejected=M` 及生成的主动学习 CSV 路径。

#### 7. 人工复核（GUI）

```bash
python manual_classify_gui.py
```

GUI 面向 `output/` 的推理结果，提供：

- 数据集与标签的流式导航按钮（自适应多列布局）
- 缩略图懒加载、缩放（10%–200%）、Ctrl+Tab / Ctrl+Shift+Tab 快速切换目录
- 普通单击切换选中、Shift 连选、双击进入独立预览窗口（方向键翻页，空格切换选择）
- 跨 `output/` 与 `cache/` 的目标数据集选择，目标子目录按首字母分组并支持跳转
- 反选、从 `output/` 移除、删除（可选回收站）、一键归档 `output/` 到 `datasets/`
- 归档会同时清理 `input/` 中同名源文件

### 配置文件说明

| 键 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dataset` | 是 | — | 单层目录名，同时用于 `datasets/`、`cache/`、`models/` |
| `seed` | 否 | `3407` | 全局随机种子；不同任务可各自设置 |
| `data.cache_long_side` | 否 | `1280` | 缓存图片最长边 |
| `data.input_size` | 否 | `256` | 模型输入分辨率 |
| `data.num_workers` | 否 | `2` | DataLoader worker 数 |
| `train.max_epochs` | 否 | `50` | 常规训练的最大 epoch |
| `train.micro_batch` | 否 | `8` | 每步前向的 batch（实际会乘 2 视图） |
| `train.grad_accum` | 否 | `4` | 梯度累积步数 |
| `train.mixed_precision` | 否 | `true` | 固定为 `true` |
| `train.early_stop_patience` | 否 | `10` | 验证指标连续不提升的容忍 epoch |
| `train.adaptation_epochs` | 否 | `12` | 数据集变更后适配周期长度 |
| `model.name` | 否 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 固定 |
| `model.lora_rank` | 否 | `16` | 固定 |
| `model.lora_alpha` | 否 | `32` | 固定 |

### 环境变量

| 变量 | 用途 |
| --- | --- |
| `HF_TOKEN` | Hugging Face 访问令牌，用于拉取 gated 的 DINOv3 仓库（与 `hf auth login` 二选一） |
| `HF_ENDPOINT` | 自定义 Hugging Face 镜像端点；未设置时使用官方默认端点 |

### 常见问题

<details>
<summary><b>训练时报错 “CUDA GPU is required”</b></summary>

本项目不支持 CPU 训练或推理，请确认已安装 CUDA 版 PyTorch，且 `torch.cuda.is_available()` 返回 `True`。
</details>

<details>
<summary><b>无法加载 DINOv3 主干</b></summary>

`facebook/dinov3-vitb16-pretrain-lvd1689m` 是 gated 仓库。请在浏览器中访问其 Hugging Face 页面并同意使用条款，然后执行 `hf auth login`，或设置 `HF_TOKEN` 后再运行一次训练以填充本地缓存。若使用镜像可设置 `HF_ENDPOINT`。
</details>

<details>
<summary><b>CUDA 显存不足（OOM）</b></summary>

将 `train.micro_batch` 从 `8` 调低（例如 `6`），然后使用 `--resume auto` 从同一检查点继续训练；`grad_accum` 建议保持 `4` 以维持有效 batch。
</details>

<details>
<summary><b>新增或删除标签后无法续训</b></summary>

- **新增标签**：允许。训练时会从当前模型的特征出发初始化新的 ArcFace 行，并从最后一个 epoch 检查点继续。
- **删除标签**：不允许。分类器与原型中的旧类状态无法安全退化，训练会以明确错误拒绝恢复；请新建一个任务配置或删除 `logs/<task>/` 后从头训练。

此外，当 `split_manifest.jsonl` 内容发生任何变化（图片新增/移除/缓存变化）时，训练会自动切换到有限 epoch 的适配周期。
</details>

<details>
<summary><b>推理结果被大量拒绝</b></summary>

拒绝由两个独立门限共同决定：全局概率阈值与类别级余弦门限。可以在 `inference.py` 中通过 `--threshold` 覆盖概率阈值以快速验证；类别级门限来源于训练集内留一相似度的 5% 分位数并截断到 `[0.45, 0.90]`，如需根本性调整，需在训练数据上补充更多该类样本后重新训练。
</details>

<details>
<summary><b>GUI 缩略图显示为空白或闪烁</b></summary>

GUI 后台任务通过 `QueuedConnection` 强制回到主线程，且解码在独立线程池中进行。若出现异常，请确认 `Pillow` 版本支持 `Image.draft`，并避免在网络盘上放置 `output/`；也可以先调低缩放比例让缩略图变小以减轻内存压力。
</details>

### License

本项目基于 MIT License 发布。

---

## English

[简体中文](#简体中文) | [English](#english)

A **DINOv3 + LoRA** image classification toolkit covering data auditing, caching, training, inference, and a human review GUI.

### Overview

This project fine-tunes the Hugging Face `facebook/dinov3-vitb16-pretrain-lvd1689m` backbone (kept frozen) via LoRA and a custom 256-d metric head combining ArcFace and supervised contrastive learning. After training, predictions are filtered by a class-specific cosine prototype rejection gate and a temperature-calibrated probability threshold, then automatically sorted into `output/<dataset>/<tag>`.

### Features

- **DINOv3 + LoRA fine-tuning**: the ViT-B/16 backbone stays frozen; only `qkv/proj/fc1/fc2` projections in Transformer blocks 9–12 are adapted (24 physical targets, rank 16 / alpha 32, locked).
- **Metric head**: `768 → 512 → 256` L2-normalized embeddings optimized with ArcFace and a supervised contrastive objective.
- **EMA evaluation**: validation uses exponential-moving-average parameters for stable metrics.
- **Expandable classifier head**: adding new tags preserves existing ArcFace rows and optimizer state; expansion only happens when needed.
- **Per-class adaptive training gate**: a class is removed from the active set once its validation accuracy reaches 95%; when all classes are satisfied, training stops early.
- **Strict checkpoints and registry**: models are published as immutable versions under `models/<dataset>/versions/`, switched atomically via `current.json`; stale versions are pruned automatically so inference always sees a complete model.
- **Reusable image cache**: hashed, deduplicated, EXIF-corrected sources are cached as lossless WebP with `cache/<dataset>/cache_index.jsonl` recording tags, so other tasks can reuse the cache even if `datasets/<dataset>` is gone.
- **Frozen split manifest**: `split_manifest.jsonl` is written once and reused. Dataset changes trigger a bounded adaptation cycle instead of corrupting the split.
- **Rejection gate + active learning**: inference applies both a global probability threshold and per-class cosine gates; rejected images are exported to `logs/<task>/active_learning/*.csv` using uncertainty prefiltering plus greedy k-center selection.
- **United inference (`__UNITEINFER__`)**: compares the maximum class probability across every registered dataset model per image.
- **Low-confidence bypass**: images whose best probability is below 0.20 are copied to `output/OUTOFTAG/OUTOFTAG` (capped at 300).
- **PySide6 review GUI**: cross-dataset reassignment, thumbnail caching, Shift range-select, preview dialog, removal, deletion, and one-click archiving.

### Tech Stack

| Area | Component |
| --- | --- |
| Language | Python 3.10+ (`X \| Y` annotations are used) |
| Deep learning | PyTorch, torchvision |
| Backbone | Hugging Face Transformers (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| Fine-tuning | PEFT (LoRA) |
| Imaging | Pillow, pillow-heif |
| Metrics | scikit-learn |
| Plotting | Matplotlib (Agg backend) |
| GUI | PySide6; optional: send2trash, pypinyin |
| Config | PyYAML |
| Misc | NumPy, tqdm |

> Training and inference strictly require a CUDA GPU (`common.require_cuda`); there is no CPU fallback.

### Project Structure

```text
.
├── configs/                       # One YAML per training task
│   ├── Tband.yaml
│   ├── Tgi1.yaml
│   ├── Thi3.yaml
│   ├── Thsr1.yaml
│   ├── Tpixiv1.yaml
│   └── Tzzz1.yaml
├── scripts/
│   ├── train.py                   # Training entry point
│   └── inference.py               # Inference entry point
├── src/
│   └── image_auto_classifier
/
│       ├── __init__.py
│       ├── common.py              # Atomic writes, RNG, logging, CUDA check
│       ├── config.py              # YAML parsing and strict validation
│       ├── data.py                # Audit, cache, splits, sampler, loaders
│       ├── model.py               # DINOv3 + LoRA + ArcFace, plus EMA
│       ├── losses.py              # ArcFace and SupCon objectives
│       ├── metrics.py             # Evaluation, calibration, prototypes, AL
│       ├── checkpointing.py       # Checkpoint schema, registry, pruning
│       ├── trainer.py             # Training loop, scheduler, early stop
│       └── inferencer.py          # Registered-model inference and exports
└── manual_classify_gui.py         # PySide6 review tool
```

Runtime layout (created on demand):

```text
datasets/<dataset>/<tag>/          # Source images
cache/<dataset>/images/<tag>/      # Lossless WebP cache + cache_index.jsonl
models/<dataset>/                  # Registry: current.json + versions/<id>/{best.ckpt, metadata.json}
logs/<task_name>/                  # Logs, split_manifest.jsonl, curves.png, models/*.ckpt, active_learning/*.csv
output/<dataset>/<tag>/            # Inference output
output/OUTOFTAG/OUTOFTAG/          # Low-confidence bypass
input/                             # Inference input directory
```

### Getting Started

#### 1. Prerequisites

- Python 3.10 or newer
- A CUDA-capable NVIDIA GPU (with a CUDA build of PyTorch)
- Access to `facebook/dinov3-vitb16-pretrain-lvd1689m`: this is a gated repository — accept its terms on Hugging Face, then authenticate with `hf auth login` or set `HF_TOKEN`

#### 2. Install

```bash
git clone <repository-url>
cd DINOv3-Image-Classifier


# Install torch / torchvision matching your CUDA version first (see https://pytorch.org/)
pip install torch torchvision
pip install transformers peft pillow pillow-heif pyyaml numpy scikit-learn matplotlib tqdm

# GUI (optional)
pip install PySide6
pip install send2trash pypinyin   # optional: recycle-bin support and CJK initial grouping
```

The project uses a `src/` layout, but `scripts/*.py` and `manual_classify_gui.py` already insert `src/` into `sys.path`, so no extra install step is required to run from the repository root.

#### 3. Prepare data

Place source images under tag directories:

```text
datasets/<dataset>/
├── tag_a/
│   ├── 001.jpg
│   └── 002.png
└── tag_b/
    └── 003.webp
```

Supported suffixes for training audits: `.png`, `.jpg`, `.jpeg`, `.heic`, `.webp`. The GUI additionally accepts `.bmp`, `.gif`, `.tif`, `.tiff`.

The first training run automatically hashes sources, detects cross-tag duplicates, corrects EXIF orientation, writes lossless WebP caches, performs a class-balanced 80/20 split, and records it in `logs/<task>/split_manifest.jsonl`.

#### 4. Configure a task

Create a YAML file under `configs/`; the file stem becomes the task name:

```yaml
dataset: band            # datasets/band; also used for output/ and models/
seed: 42
data:
  cache_long_side: 1280  # Long side of the cache; larger images are downscaled
  input_size: 256        # Model input resolution
  num_workers: 2
train:
  max_epochs: 50
  micro_batch: 8
  grad_accum: 4
  mixed_precision: true  # locked to true
  early_stop_patience: 10
  adaptation_epochs: 12  # Length of the post-dataset-change adaptation cycle
model:
  name: facebook/dinov3-vitb16-pretrain-lvd1689m
  lora_rank: 16          # locked
  lora_alpha: 32         # locked
```

`model.name`, `model.lora_rank`, `model.lora_alpha`, and `train.mixed_precision` are locked; changing them raises `ConfigurationError`. Six example configurations ship in `configs/`.

#### 5. Train

```bash
python scripts/train.py --config configs/Tband.yaml

# Resume from the latest epoch checkpoint of this task
python scripts/train.py --config configs/Tband.yaml --resume auto
```

Training highlights:

- LR schedule: 3-epoch linear warm-up, then cosine decay to `1e-6`.
- LoRA stays frozen for the first 3 epochs; from epoch 4 it becomes trainable.
- Every epoch the EMA parameters produce validation macro-F1; if the metric fails to improve by ≥ 0.002 for `early_stop_patience` epochs, training stops.
- Each epoch writes `logs/<task>/models/epoch_NNN.ckpt` (only the latest 3 are kept). On validation improvement, the epoch checkpoint is published to `models/<dataset>/versions/<id>/` and `current.json` is updated atomically.
- If the `split_manifest.jsonl` hash changes, a bounded `adaptation_epochs` cycle starts from `logs/<task>/models/best.ckpt`. Added tags expand ArcFace; removed tags cause a hard failure.
- Resume-friendly: optimizer, scheduler, GradScaler, and RNG states are all stored in checkpoints.

#### 6. Inference

```bash
# Use every registered dataset model
python scripts/inference.py --dataset __UNITEINFER__ --input input

# Use a single registered dataset model
python scripts/inference.py --dataset band --input input

# Override the calibrated probability threshold
python scripts/inference.py --dataset band --input input --threshold 0.9
```

The pipeline:

1. Runs EMA forward passes to obtain 256-d embeddings and logits.
2. Looks up the class-specific prototype, computes cosine similarity, and compares it with the class gate.
3. Also filters using the temperature-calibrated softmax probability against the probability threshold.
4. Copies images that pass both gates into `output/<dataset>/<tag>/`. Rejected images are exported to `logs/<task>/active_learning/<timestamp>.csv` (up to 200).
5. Copies images with probability below `0.20` into `output/OUTOFTAG/OUTOFTAG/` (up to 300).
6. Prints `accepted=N rejected=M` and any generated active-learning CSV paths.

#### 7. Human review (GUI)

```bash
python manual_classify_gui.py
```

The GUI operates on `output/` inference results and provides:

- Flow-layout dataset and tag navigation that adapts to available width
- Lazy-loaded, zoomable (10%–200%) thumbnails; Ctrl+Tab / Ctrl+Shift+Tab to switch directories
- Plain click toggles selection, Shift-click selects a range, double-click opens a preview dialog (arrow keys to navigate, Space to toggle)
- Target dataset selection across `output/` and `cache/`, with subdirectories grouped by initial letter and jump buttons
- Invert selection, remove from `output/`, delete (optionally to recycle bin), and one-click archiving of `output/` into `datasets/`
- Archiving also cleans up matching source files under `input/`

### Configuration Reference

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `dataset` | yes | — | Single directory name; used across `datasets/`, `cache/`, `models/` |
| `seed` | no | `3407` | Global random seed; each task may override |
| `data.cache_long_side` | no | `1280` | Long side of the cache |
| `data.input_size` | no | `256` | Model input resolution |
| `data.num_workers` | no | `2` | DataLoader workers |
| `train.max_epochs` | no | `50` | Maximum epochs in a normal run |
| `train.micro_batch` | no | `8` | Per-step forward batch (each source produces two views) |
| `train.grad_accum` | no | `4` | Gradient accumulation steps |
| `train.mixed_precision` | no | `true` | Locked to `true` |
| `train.early_stop_patience` | no | `10` | Non-improving epochs tolerated |
| `train.adaptation_epochs` | no | `12` | Length of the adaptation cycle |
| `model.name` | no | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Locked |
| `model.lora_rank` | no | `16` | Locked |
| `model.lora_alpha` | no | `32` | Locked |

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Hugging Face access token for the gated DINOv3 repository (alternative to `hf auth login`) |
| `HF_ENDPOINT` | Custom Hugging Face endpoint; when unset, the official endpoint is used |

### Troubleshooting

<details>
<summary><b>"CUDA GPU is required" during training</b></summary>

The project does not support CPU training or inference. Confirm the CUDA build of PyTorch is installed and that `torch.cuda.is_available()` returns `True`.
</details>

<details>
<summary><b>Cannot load the DINOv3 backbone</b></summary>

`facebook/dinov3-vitb16-pretrain-lvd1689m` is a gated repository. Open its Hugging Face page, accept the access terms, then run `hf auth login` or set `HF_TOKEN` and rerun training once to populate the local cache. If you use a mirror, set `HF_ENDPOINT` accordingly.
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

Lower `train.micro_batch` (for example from `8` to `6`) and resume from the same checkpoint with `--resume auto`. Keep `grad_accum` at `4` to preserve the effective batch size.
</details>

<details>
<summary><b>Cannot resume after tag changes</b></summary>

- **Added tags**: allowed. New ArcFace rows are seeded from the current model's features and training continues from the last epoch checkpoint.
- **Removed tags**: not allowed. Legacy classifier/prototype state cannot be safely shrunk; training fails with a clear error. Create a new task configuration or delete `logs/<task>/` and train from scratch.

Additionally, any change to `split_manifest.jsonl` (new images, removed images, cache changes) automatically triggers a bounded adaptation cycle.
</details>

<details>
<summary><b>Many images are rejected during inference</b></summary>

Rejection is decided by two independent gates: a global probability threshold and per-class cosine gates. You can quickly override the probability threshold with `--threshold` in `inference.py`. Class gates come from the 5th percentile of leave-one-out similarities on the training set, clamped to `[0.45, 0.90]`; to change them fundamentally, add more training samples for that class and retrain.
</details>

<details>
<summary><b>GUI thumbnails are blank or flicker</b></summary>

Background work is forced back to the main thread via `QueuedConnection`, and decoding happens on a worker pool. If issues persist, verify that your `Pillow` build supports `Image.draft`, avoid placing `output/` on a network drive, and lower the zoom level to reduce memory pressure.
</details>

### License

Released under the MIT License.
