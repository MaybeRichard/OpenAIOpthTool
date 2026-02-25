# OpenAIOpthTool

An open-source, ophthalmologist-friendly AI toolkit that integrates multiple ophthalmic image analysis models into a unified, easy-to-use platform.

一个开源的、面向眼科医生的多模型AI工具平台，将多种眼科影像分析模型整合为统一、易用的工具。

## Features

- **Multi-modality**: Supports fundus photographs, OCT images, and more (expanding)
- **Multi-task**: Classification, segmentation, denoising, and beyond
- **User-friendly**: Designed for clinicians with no AI/programming background
- **CPU & GPU**: Automatically detects hardware and runs on both CPU and GPU
- **Extensible**: New models can be easily integrated through a unified interface

## Available Models

| Task | Model | Modality | Description | Reference |
|------|-------|----------|-------------|-----------|
| Classification | ResNet-18 | Fundus | Myopic maculopathy grading (5 META-PM classes) | [Yii et al., MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-54857-4_8) |
| Segmentation | UNet (ResNet-34) | Fundus | Optic disc & cup segmentation (REFUGE2) | - |
| Segmentation | TransUNet (R50-ViT-B/16) | Fundus | Optic disc & cup segmentation (REFUGE2) | [Chen et al., 2021](https://arxiv.org/abs/2102.04306) |

## Quick Start

### Requirements

- Python >= 3.8
- PyTorch >= 1.10

```bash
pip install torch torchvision numpy opencv-python pandas scikit-learn pillow matplotlib tqdm
pip install segmentation_models_pytorch ml_collections scipy  # for segmentation models
```

### Unified API (Recommended)

All models can be accessed through a single unified interface:

```python
from opthtools import registry

# List all available models
for m in registry.list_models():
    print(f"{m['name']:40s} {m['task']}")

# Classification: Myopic Maculopathy Grading
model = registry.load("myopic_maculopathy_resnet18",
                       checkpoint_path="models/Classification/MyopicMaculopathyClassification/weights/bestModel.pth")
result = model.predict("path/to/fundus_image.jpg")
print(f"Class: {result.pred_class} - {result.class_name}")
print(f"Probabilities: {result.probabilities}")

# Segmentation: Optic Disc & Cup
model = registry.load("optic_disc_cup_unet",
                       checkpoint_path="models/segmentation/segmentation_unet_refuge2/best.pth")
result = model.predict("path/to/fundus_image.jpg")
# result.mask: same size as original image, class indices (0=background, 1=cup, 2=disc)
# result.probabilities: per-class probability maps
```

The `predict()` method accepts file paths, PIL Images, or numpy arrays. Device is auto-detected (CUDA > MPS > CPU) or manually specified via `device="cpu"`.

## Project Structure

```
OpenAIOpthTool/
├── opthtools/                                   # Unified API layer
│   ├── __init__.py                              # Public API
│   ├── base.py                                  # BaseModel ABC + result dataclasses
│   ├── registry.py                              # Model registry
│   ├── device.py                                # Auto device selection
│   ├── utils.py                                 # Image loading utilities
│   └── adapters/                                # One file per model
│       ├── classification_resnet18.py
│       ├── segmentation_unet.py
│       └── segmentation_transunet.py
├── models/                                      # Raw model implementations
│   ├── Classification/
│   │   └── MyopicMaculopathyClassification/
│   └── segmentation/
│       ├── train_unet.py / train_transunet.py
│       ├── evaluate.py
│       ├── datasets/
│       └── models/transunet/
├── run.sh
├── CLAUDE.md
├── LICENSE
└── README.md
```

## Roadmap

- [x] Fundus classification (myopic maculopathy)
- [x] Fundus segmentation (optic disc & cup)
- [x] Unified model registry and inference API
- [ ] Web interface for interactive use
- [ ] Desktop application
- [ ] OCT image analysis models
- [ ] Denoising models
- [ ] Batch processing support

## Citation

If you use this toolkit in your research, please cite the relevant papers for each model:

```bibtex
@inproceedings{yii2024clinically,
  title={A Clinically Guided Approach for Training Deep Neural Networks for Myopic Maculopathy Classification},
  author={Yii, Fabian},
  booktitle={Myopic Maculopathy Analysis. MICCAI 2023. Lecture Notes in Computer Science},
  volume={14563},
  year={2024},
  publisher={Springer, Cham}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have ophthalmic AI models (classification, segmentation, denoising, etc.) that you would like to integrate, please open an issue or submit a pull request.
