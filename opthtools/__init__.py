"""
OpenAIOpthTool - 眼科AI工具统一接口 / Unified Ophthalmic AI Toolkit API

Usage:
    from opthtools import registry

    # 查看可用模型 / List available models
    registry.list_models()

    # 加载并预测 / Load and predict
    model = registry.load("myopic_maculopathy_resnet18",
                          checkpoint_path="models/.../bestModel.pth")
    result = model.predict("path/to/fundus_image.jpg")
"""

from .registry import registry
from .base import (
    BaseModel,
    ClassificationResult,
    SegmentationResult,
    PredictionResult,
)
from .device import get_device

# Import adapters to trigger registration
from .adapters import classification_resnet18  # noqa: F401
from .adapters import segmentation_unet  # noqa: F401
from .adapters import segmentation_transunet  # noqa: F401

__all__ = [
    "registry",
    "BaseModel",
    "ClassificationResult",
    "SegmentationResult",
    "PredictionResult",
    "get_device",
]
