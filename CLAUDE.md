# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

OpenAIOpthTool 的最终目标是构建一个**开源的、面向眼科医生的多模型AI工具平台**。核心设计原则：

1. **非专业用户友好**：眼科医生无需AI/编程背景即可使用，界面简洁直观
2. **多模态支持**：覆盖OCT、Fundus等多种眼科影像类型
3. **多任务支持**：分割（Segmentation）、分类（Classification）、去噪（Denoising）等
4. **多种使用方式**：Python API / Web界面 / 桌面程序，逐步实现
5. **CPU与GPU兼容**：自动检测硬件环境，均可运行
6. **可扩展架构**：新模型可通过统一接口轻松接入，不需要修改核心代码

当前阶段已有的模型作为第一批接入的算法，后续将持续收集和集成更多眼科AI模型。所有开发工作都应围绕这一目标进行，确保架构设计的前瞻性和可扩展性。

## Unified API (`opthtools/`)

统一模型注册与推理接口，用户 3 行代码即可调用任何模型：

```python
from opthtools import registry
model = registry.load("myopic_maculopathy_resnet18", checkpoint_path="models/.../bestModel.pth")
result = model.predict("fundus.jpg")
```

### Architecture

- `opthtools/base.py`: `BaseModel` ABC + `ClassificationResult` / `SegmentationResult` dataclasses
- `opthtools/registry.py`: `ModelRegistry` — `@registry.register` 装饰器注册，`list_models()` 列表，`load()` 实例化
- `opthtools/device.py`: `get_device()` 自动选择 CUDA > MPS > CPU
- `opthtools/utils.py`: `load_image()` 统一输入（str/PIL/ndarray → PIL RGB）
- `opthtools/adapters/`: 每个模型一个文件，继承 `BaseModel`，实现 `_load_model()` 和 `predict()`

### Adding a new model

1. 在 `opthtools/adapters/` 下新建文件，继承 `BaseModel`，加 `@registry.register`
2. 在 `opthtools/__init__.py` 加一行 import

### Registered models

| Name | Task | Modality |
|------|------|----------|
| `myopic_maculopathy_resnet18` | classification | fundus |
| `optic_disc_cup_unet` | segmentation | fundus |
| `optic_disc_cup_transunet` | segmentation | fundus |

## Raw Model Details

### Models directory

- **Classification** (`models/Classification/MyopicMaculopathyClassification/`): Myopic maculopathy grading (5 META-PM classes) using ResNet-18. MICCAI MMAC 2023.
- **Segmentation** (`models/segmentation/`): UNet + TransUNet on REFUGE2 (optic disc/cup, 3-class).

### Training commands

#### Classification (run from `models/Classification/MyopicMaculopathyClassification/code/`)

```bash
# Train
python train.py --dataDir ../data --weightDir ../weights --num_epochs 50 --batch_size 20 --lr 5e-5

# Inference (standalone, without opthtools)
python inference.py --image <path_to_image> --weights ../weights/bestModel.pth
```

#### Segmentation (run from `models/segmentation/`)

```bash
# Install dependencies
pip install segmentation_models_pytorch ml_collections scipy

# Download TransUNet pretrained weights
mkdir -p pretrained && wget -O pretrained/R50+ViT-B_16.npz https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz

# Train UNet (single dataset)
python train_unet.py --dataset refuge2 --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4

# Train TransUNet (single dataset)
python train_transunet.py --dataset refuge2 --gpu 0 --epochs 150 --batch_size 24 --lr 0.01

# Evaluate all models on all datasets
python evaluate.py --dataset all --model all --gpu 0
```

See `run_segmentation.sh` for the full multi-GPU parallel training commands.

## Architecture

### Classification Pipeline
1. **Image Synthesis** (`code/imageSynthesis.ipynb`, `code/utils.py:imageSynthesiser`): Generates synthetic fundus images by compositing lesion masks onto backgrounds
2. **Training** (`code/train.py`): ResNet-18 with mixup augmentation (p=0.5, alpha=0.4), weighted CrossEntropyLoss with label smoothing=0.1, Adam optimizer, cosine annealing LR. Images resized to 512x512, normalized to [0,1]
3. **Inference** (`code/inference.py`, `code/model.py`): Test-Time Augmentation (TTA) with 11 variants (original + 2 flips + 8 rotations at +/-5, +/-8, +/-12, +/-15 degrees), averaged softmax scores

Key classes: `ResNet18` (model), `trainedModel`/`InferenceModel` (inference wrappers), `fundusDataset` (dataset), `augmentation`/`imageSynthesiser` (utils)

### Segmentation Pipeline
- **UNet**: `segmentation_models_pytorch.Unet` with ResNet-34 ImageNet-pretrained encoder
- **TransUNet**: Custom R50-ViT-B/16 implementation (`models/transunet/`) with ImageNet-21k pretrained weights loaded from `.npz`. Uses weight-standardized Conv2d + GroupNorm (not BatchNorm) in the ResNetV2 backbone
- **Loss**: Combined BCE+Dice (`BCEDiceLoss`) for binary tasks, CE+Dice (`CEDiceLoss`) for multi-class
- **Metrics**: Per-sample Dice and IoU computed in `metrics.py`. Multi-class metrics skip background (class 0)
- **Dataset**: `SegmentationDataset` supports cvc/kvasir/refuge2 with 90/10 train/val split (seed=42), ImageNet normalization, resolution 224x224

Checkpoints saved to `checkpoints/{model}_{dataset}/best.pth`.

## Important Notes

- PyTorch >= 2.6 requires `weights_only=False` in `torch.load()`. The `inference.py` handles this; `model.py:trainedModel.load()` does NOT (known issue).
- Classification uses `cv2.imread` (BGR order); segmentation uses `PIL.Image.open` (RGB order).
- Segmentation dataset paths in `seg_dataset.py:DATASET_CONFIGS` are hardcoded to `/data2/sichengli/Data/test/Segmentation/...` — must be updated for local environments.
- Classification `train.py` argparse has `betas1`/`betas2` typed as `int` instead of `float` (known bug).
- The classification data CSV (`combinedTrainingLabels.csv`) rows: 1-1143 MMAC, 1144-2343 PALM, 2344-2843 synthesized. The `get_combined_df()` util filters out problematic synthesized entries.
