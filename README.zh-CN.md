# Image Auto Classifier

[English](README.md) | [简体中文](README.zh-CN.md)

Image Auto Classifier 是一个基于 **DINOv3 ViT-B/16 + LoRA** 的本地图像自动分类与数据集维护项目。

本项目为不同数据集分别训练分类模型，将最佳模型注册用于推理，通过概率阈值和类别原型相似度过滤不可靠预测，并将不确定样本导出到主动学习列表。推理完成后，还可以使用 Tkinter 图形界面对分类结果进行人工复核、纠正、删除和归档。

整体工作流围绕本地目录完成：

```text
datasets → cache → 训练 → 注册模型
                         ↓
input → 推理 → output → 人工复核 → datasets
                    ↘ 主动学习
```

## 核心特性

* 使用 DINOv3 ViT-B/16 作为图像编码器，冻结主干参数并通过 LoRA 进行参数高效微调。
* 在 Transformer 第 9–12 个 Block 的 Attention 与 MLP 投影层应用 LoRA。
* `768 → 512 → 256` 的归一化度量学习 Head。
* ArcFace 分类目标与监督式对比学习联合训练。
* 使用 EMA 参数进行评估和推理。
* 确定性训练以及可恢复的 Epoch Checkpoint。
* 基于 SHA-256 内容追踪的持久化图像缓存。
* 跨训练运行保持稳定的训练集 / 验证集划分清单。
* 数据集新增图片或类别后自动进入适配训练流程。
* Temperature Calibration 与自动概率阈值选择。
* 基于类别原型的独立余弦相似度拒绝阈值。
* 支持单数据集模型推理以及多模型统一推理。
* 自动导出被拒绝样本的主动学习 CSV。
* 原子化最佳模型注册，避免推理读取到不完整模型。
* Tkinter 人工复核与数据集整理界面。
* 训练和推理支持 HEIC、JPEG、PNG 与 WebP。

## 技术栈

| 组件       | 技术                                         |
| -------- | ------------------------------------------ |
| 编程语言     | Python 3.10+                               |
| 深度学习     | PyTorch、torchvision                        |
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| 模型加载     | Hugging Face Transformers                  |
| 参数高效微调   | PEFT / LoRA                                |
| 图像处理     | Pillow、pillow-heif                         |
| 评估指标     | scikit-learn                               |
| 配置       | PyYAML                                     |
| 训练进度     | tqdm                                       |
| 训练曲线     | Matplotlib                                 |
| 人工复核界面   | Tkinter                                    |
| 数值计算     | NumPy                                      |

GUI 在安装 `send2trash` 时支持移动到系统回收站；安装 `pypinyin` 后，可以按照拼音首字母对非拉丁字符标签进行分组。

## 项目架构

### 训练流程

训练从一份 YAML 任务配置开始：

```text
配置
 │
 ├─ 数据集审计
 │    ├─ 图像有效性检查
 │    ├─ SHA-256 去重
 │    ├─ 无损 WebP 缓存
 │    └─ 持久化数据划分清单
 │
 ├─ DINOv3 ViT-B/16
 │    └─ Block 9–12 LoRA
 │
 ├─ 768 → 512 → 256 度量学习 Head
 │
 ├─ ArcFace + 监督式对比学习
 │
 ├─ EMA 评估
 │
 ├─ 概率校准 + 类别原型
 │
 └─ Checkpoint + 最佳模型注册
```

DINOv3 Backbone 保持冻结状态，仅训练 LoRA 参数、度量学习 Head 和 ArcFace 分类器。

每个训练样本会生成两份增强视图，联合优化：

```text
0.75 × ArcFace Cross Entropy
+ 0.25 × Supervised Contrastive Loss
```

LoRA 参数会在调度的前三个 Epoch 之后启用。优化器分别为 LoRA 与 Head 使用不同学习率，并采用三个 Epoch 的线性 Warm-up，之后进行余弦衰减。

### 推理流程

对于每张输入图片，推理阶段会计算：

1. 预测类别；
2. 校准后的最大类别概率；
3. 图片特征与预测类别原型之间的余弦相似度。

只有同时满足概率阈值和该类别对应的相似度阈值，结果才会被接受。

接受的图片会复制到：

```text
output/<dataset>/<tag>/
```

被拒绝的图片可以进入主动学习流程，供后续人工标注。

### 模型注册机制

每个数据集当前使用的模型由以下结构管理：

```text
models/<dataset>/
├── current.json
└── versions/
    └── <registry-version>/
        ├── best.ckpt
        └── metadata.json
```

发布新的最佳模型时，会先创建完整且不可变的版本目录，然后原子更新 `current.json` 指向新版本，因此并发推理不会读取到只写入一部分的模型。

完成切换后，旧注册版本会被清理，因此模型注册目录保留当前激活的最佳模型，而不是历史回滚版本。

## 项目结构

```text
.
├── configs/
│   └── task1.yaml
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

核心模块职责：

| 模块                       | 职责                                       |
| ------------------------ | ---------------------------------------- |
| `config.py`              | YAML 配置读取、默认值以及严格参数校验                    |
| `data.py`                | 数据集审计、缓存、划分清单、数据增强、采样与 DataLoader        |
| `model.py`               | DINOv3 加载、精确 LoRA 注入、度量 Head、ArcFace、EMA |
| `losses.py`              | ArcFace 层与监督式对比损失                        |
| `metrics.py`             | 模型评估、概率校准、类别原型、相似度阈值、主动学习                |
| `trainer.py`             | 完整训练流程、恢复、数据变化适配、评估与提前停止                 |
| `checkpointing.py`       | Checkpoint Schema、原子化保存以及最佳模型注册          |
| `inferencer.py`          | 注册模型加载、推理、拒绝机制与结果复制                      |
| `manual_classify_gui.py` | 人工复核、纠正、删除和归档                            |

## 环境要求

### Python

项目使用了要求 **Python 3.10 或更高版本** 的 Python 语法和标准库能力。

### NVIDIA CUDA GPU

训练和推理入口都会显式检查 CUDA：

```text
CUDA GPU is required
```

当前训练与推理流程不支持纯 CPU 运行。

CUDA 混合精度训练也是固定训练方案的一部分。

### DINOv3 模型权限

本项目固定使用：

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

该 Hugging Face 仓库需要申请访问权限。

第一次训练前，需要：

1. 在 Hugging Face 上申请 / 接受对应模型的访问条款；
2. 在运行项目的机器上完成认证：

```bash
hf auth login
```

也可以通过环境变量提供 Hugging Face Token：

```bash
HF_TOKEN=<your-token>
```

项目默认使用 Hugging Face 标准服务端点。如果需要使用其他端点，可以配置：

```bash
HF_ENDPOINT=<endpoint>
```

至少需要成功运行一次训练，使 DINOv3 模型文件进入本地 Hugging Face Cache。注册模型推理会使用本地缓存重新构建模型结构。

## 数据集格式

训练数据统一放在：

```text
datasets/<dataset>/
```

每个一级子目录代表一个分类标签：

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

训练图片需要直接放置在对应标签目录中。

支持的源图片格式：

```text
.png
.jpg
.jpeg
.heic
.webp
```

数据审计阶段会解码图片、应用 EXIF 方向信息，并将有效图片保存为无损 WebP 缓存。

缓存结构如下：

```text
cache/<dataset>/
├── cache_index.jsonl
└── images/
    └── <tag>/
        └── <sha256>.webp
```

`cache_index.jsonl` 持久化保存标签和内容信息，因此缓存数据可以独立于原始数据集目录继续使用。

同一个标签下内容完全相同的图片会去重；如果相同 SHA-256 内容同时出现在两个不同标签中，则视为标签冲突并停止数据审计。

## 配置

每个训练任务由 `configs/` 下的一份 YAML 文件定义。

例如：

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

配置文件名称同时决定训练任务名称。

例如：

```text
configs/task1.yaml
```

对应任务名称：

```text
task1
```

其中 `dataset` 字段决定实际使用的数据目录：

```text
datasets/task1/
```

### 配置项说明

| 配置                          |             默认值 | 说明                   |
| --------------------------- | --------------: | -------------------- |
| `dataset`                   |              必填 | 数据集目录名以及模型注册名称       |
| `seed`                      |          `3407` | 全局确定性随机种子            |
| `data.cache_long_side`      |          `1280` | 缓存图片最长边              |
| `data.input_size`           |           `256` | 模型输入尺寸               |
| `data.num_workers`          |             `2` | DataLoader Worker 数量 |
| `train.max_epochs`          |            `50` | 常规训练最大 Epoch         |
| `train.micro_batch`         |             `8` | Micro Batch 大小       |
| `train.grad_accum`          |             `4` | 梯度累积次数               |
| `train.mixed_precision`     |          `true` | CUDA AMP，固定为 `true`  |
| `train.early_stop_patience` |            `10` | 验证指标无提升的容忍 Epoch     |
| `train.adaptation_epochs`   |            `12` | 数据变化后的最大适配训练 Epoch   |
| `model.name`                | DINOv3 ViT-B/16 | 固定 Backbone          |
| `model.lora_rank`           |            `16` | 固定 LoRA Rank         |
| `model.lora_alpha`          |            `32` | 固定 LoRA Alpha        |

模型名称、LoRA Rank、LoRA Alpha 和混合精度设置属于固定训练方案，尝试修改为其他值会在配置解析阶段被拒绝。

## 训练

以下命令均在项目根目录执行。

训练一个任务：

```bash
python scripts/train.py --config configs/task1.yaml
```

项目中已经提供的其他任务配置可以使用相同方式运行：

```bash
python scripts/train.py --config configs/task1.yaml
```

### 恢复训练

从当前任务最新的 Epoch Checkpoint 恢复：

```bash
python scripts/train.py \
  --config configs/task1.yaml \
  --resume auto
```

恢复时会校验已保存的数据集配置、预处理参数、模型结构、LoRA 配置、优化器、学习率调度器、GradScaler 和随机数状态。

恢复训练期间不允许删除已有标签，因为这会使已保存的分类器和类别原型状态失效。

当数据集新增图片或类别时，训练会进入一个有上限的适配周期。已有 Class ID 会保持不变，新类别会依次追加。

### 训练产物

以 `task1` 为例，任务产物写入：

```text
logs/task1/
├── task1.log
├── curves.png
├── split_manifest.jsonl
└── models/
    ├── best.ckpt
    └── epoch_XXX.ckpt
```

Epoch Checkpoint 最多保留最近三个。

`split_manifest.jsonl` 会持久化数据集划分。后续增加图片时，已有样本不会重新随机分配训练集或验证集。

存在验证集时，模型选择指标为 Macro F1。某个类别验证准确率达到 95% 后，该类别会停止参与后续训练采样；所有类别都达到该阈值后，训练可以提前结束。

## 推理

推理只读取指定输入目录**第一层**中的图片。

默认输入目录：

```text
input/
```

### 使用单个数据集模型

```bash
python scripts/inference.py \
  --dataset task1 \
  --input input
```

接受的图片会复制到：

```text
output/task1/<预测标签>/
```

### 使用所有已注册模型统一推理

省略 `--dataset` 时默认进入统一推理模式：

```bash
python scripts/inference.py --input input
```

等价于：

```bash
python scripts/inference.py \
  --dataset __UNITEINFER__ \
  --input input
```

统一推理会依次加载所有已注册的数据集模型。

对于每张图片，会比较不同模型中的最大类别概率，并使用其中概率最高的类别作为候选预测；最终是否接受该图片，仍由胜出模型自身的概率阈值和类别原型相似度阈值决定。

不同 DINOv3 模型会逐个加载和释放，避免同时占用 GPU 显存。

### 覆盖概率阈值

正常情况下，推理使用训练过程中自动校准并注册的概率阈值。

可以手动覆盖：

```bash
python scripts/inference.py \
  --dataset task1 \
  --input input \
  --threshold 0.95
```

阈值必须位于 `0` 到 `1` 之间。

覆盖概率阈值不会关闭类别对应的余弦相似度过滤。

### 主动学习

被拒绝的样本可以导出到：

```text
logs/<task>/active_learning/<timestamp>.csv
```

CSV 包含：

```text
rank
u
pmax
similarity
suggested_tag
path
```

主动学习选择器会先按照不确定度筛选候选集，再结合已标注样本的特征执行 Greedy k-center，以兼顾不确定度和样本多样性，最终最多选择 200 张图片。

## 人工复核 GUI

在项目根目录运行：

```bash
python manual_classify_gui.py
```

GUI 主要处理：

```text
output/<dataset>/<tag>/
```

中的推理结果，支持：

* 数据集和预测标签目录导航；
* 响应式缩略图浏览；
* 多选与反选；
* 原图预览；
* 独立选择目标数据集；
* 按首字母导航目标标签；
* 创建新的目标标签目录；
* 在不同 output 标签 / 数据集之间移动图片；
* 仅从 `output` 移除图片；
* 同时删除 `output` 图片和 `input` 中匹配的源文件；
* 安装 `send2trash` 后删除到系统回收站；
* 将人工复核后的全部 output 递归归档回 `datasets`。

常用快捷键：

```text
Ctrl+Tab        切换到下一个 output 标签目录
Ctrl+Shift+Tab  切换到上一个 output 标签目录
```

### 推荐复核流程

完整流程可以按照以下方式进行：

```text
1. 执行推理
2. 启动 manual_classify_gui.py
3. 检查 output/<dataset>/<tag>
4. 将误分类图片移动到正确的数据集 / 标签
5. 移除或删除不需要的图片
6. 归档已经完成复核的 output
```

执行“归档全部 output”时，会递归地把：

```text
output/<dataset>/
```

合并到：

```text
datasets/<dataset>/
```

发生同名文件冲突时，不会覆盖已有文件，而是生成唯一的新文件名并保留双方内容。

只有全部待归档 output 文件都成功移动后，GUI 才会清理 `input` 中对应的同名源文件。

## 运行时目录

项目运行过程中使用以下目录：

```text
datasets/   已标注的训练数据
cache/      持久化图片解码缓存
input/      等待推理的图片
output/     已接受的推理结果以及人工复核工作区
logs/       日志、划分清单、训练曲线、Checkpoint、主动学习 CSV
models/     推理使用的已注册最佳模型
```

这些运行时目录与 Python 源码目录相互独立，并由训练、推理和人工复核流程按需创建。

## License

本项目使用 MIT License。
