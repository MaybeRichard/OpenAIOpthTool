# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Myopic Maculopathy Classification from the 2023 MICCAI Myopic Maculopathy Analysis Challenge (MMAC). Classifies fundus images into 5 META-PM categories (0-4: Normal → Tessellated fundus → Diffuse atrophy → Patchy atrophy → Macular atrophy) using ResNet-18 with image synthesis, mixup augmentation, and test-time augmentation. Ranked 1st in validation phase, 5th in test phase.

## Commands

### Training
```bash
# Default training (expects data in ~/data)
python code/train.py

# Custom configuration
python code/train.py --dataDir /path/to/data --weightDir /path/to/weights --batch_size 32 --num_epochs 100 --lr 1e-4
```

### Inference
```python
from code.model import trainedModel
import cv2 as cv

model = trainedModel(checkpoint="bestModel.pth")
model.load(dir_path="./weights")
image = cv.imread("fundus.png")
prediction = model.predict(image)  # Returns 0-4
```

### Image Synthesis Tutorial
```bash
jupyter notebook code/imageSynthesis.ipynb
```

## Architecture

### Pipeline Flow
1. **Image Synthesis** (`utils.py:imageSynthesiser`) - Generates synthetic rare-class examples by compositing lesion masks onto background fundus images
2. **Training** (`train.py`) - ResNet-18 with weighted CrossEntropy, label smoothing (0.1), mixup augmentation (α=0.4), cosine annealing
3. **Inference** (`model.py:trainedModel`) - Test-time augmentation averaging 11 variants (original + flips + 8 rotations)

### Key Classes
- `ResNet18` (`model.py`) - Pretrained torchvision ResNet-18 with FC layer adapted for 5 classes
- `trainedModel` (`model.py`) - Inference wrapper with TTA (horizontal/vertical flips + rotations at ±5°, ±8°, ±12°, ±15°)
- `fundusDataset` (`dataset.py`) - PyTorch Dataset, resizes to 512×512, returns one-hot labels [B,5]
- `augmentation` (`utils.py`) - Random rotation, horizontal flip, brightness/saturation jitter
- `imageSynthesiser` (`utils.py`) - Lesion mask extraction and composition for data augmentation

### Data Structure
```
data/
├── Images/training/          # Fundus images (PNG)
└── Groundtruths/
    └── combinedTrainingLabels.csv  # Columns: image, grade (0-4), fovea_x, fovea_y
```
CSV contains: MMAC (rows 1-1143) + PALM (rows 1144-2343) + Synthesized (rows 2344-2843, but 250 PALM-background ones excluded by `get_combined_df`)

### Lesion Bank
```
lesionBank/
├── MA_masked_MMAC/     # Macular atrophy masks (MMAC)
├── MA_masked_PALM/     # Macular atrophy masks (PALM)
├── patchy_masked_MMAC/ # Patchy atrophy masks (MMAC)
└── patchy_masked_PALM/ # Patchy atrophy masks (PALM)
```

## Training Hyperparameters (Defaults)
- Optimizer: Adam (lr=5e-5, weight_decay=5e-4, betas=(0.9, 0.999))
- Loss: Weighted CrossEntropy with label smoothing (ε=0.1)
- Scheduler: Cosine annealing
- Augmentation: Rotation ±30°, horizontal flip, color jitter, mixup (p=0.5, α=0.4)
- Epochs: 50, Batch size: 20, Image size: 512×512

## Dependencies
```
torch torchvision numpy opencv-python pandas scikit-learn pillow matplotlib tqdm
```

## PyTorch Compatibility
For PyTorch >= 2.6, add `weights_only=False` to `torch.load()` calls to avoid warnings (see `model.py:31`).
