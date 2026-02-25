from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image


@dataclass
class ClassificationResult:
    """分类模型的预测结果 / Classification prediction result."""
    pred_class: int
    class_name: str
    probabilities: list
    class_names: dict


@dataclass
class SegmentationResult:
    """分割模型的预测结果 / Segmentation prediction result."""
    mask: np.ndarray            # [H, W] class indices, same size as original image
    probabilities: np.ndarray   # [C, H, W] per-class probabilities
    num_classes: int
    class_names: dict
    original_size: tuple        # (H, W)


PredictionResult = Union[ClassificationResult, SegmentationResult]


class BaseModel(ABC):
    """所有模型适配器的基类 / Base class for all model adapters."""

    name: str = ""
    display_name: str = ""
    task: str = ""
    modality: str = ""
    description: str = ""

    def __init__(self, checkpoint_path, device=None):
        from .device import get_device
        self.checkpoint_path = checkpoint_path
        self.device = get_device(device)
        self._model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load model architecture and weights. Set self._model."""
        ...

    @abstractmethod
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            image: File path (str), PIL Image, or numpy array.

        Returns:
            ClassificationResult or SegmentationResult.
        """
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' task='{self.task}' device='{self.device}'>"
