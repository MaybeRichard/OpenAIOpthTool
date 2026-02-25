import numpy as np
import cv2 as cv
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import rotate

from ..base import BaseModel, ClassificationResult
from ..registry import registry
from ..utils import load_image

CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "豹纹状眼底 (Tessellated fundus)",
    2: "弥漫性脉络膜视网膜萎缩 (Diffuse chorioretinal atrophy)",
    3: "斑片状脉络膜视网膜萎缩 (Patchy chorioretinal atrophy)",
    4: "黄斑萎缩 (Macular atrophy)",
}


class _ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        return self.resnet(x)


@registry.register
class MyopicMaculopathyResNet18(BaseModel):
    name = "myopic_maculopathy_resnet18"
    display_name = "病理性近视黄斑病变分类 / Myopic Maculopathy Classification (ResNet-18)"
    task = "classification"
    modality = "fundus"
    description = (
        "5-class META-PM grading of myopic maculopathy from fundus photographs. "
        "Uses TTA (11 variants). MICCAI MMAC 2023."
    )

    def _load_model(self):
        self._model = _ResNet18(num_classes=5)
        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

        self._hor_flip = transforms.RandomHorizontalFlip(p=1)
        self._ver_flip = transforms.RandomVerticalFlip(p=1)

    def predict(self, image) -> ClassificationResult:
        pil_image = load_image(image)

        # RGB -> BGR to match original training pipeline (cv2.imread)
        img_bgr = np.array(pil_image)[:, :, ::-1]
        img_bgr = np.ascontiguousarray(img_bgr)
        img_bgr = cv.resize(img_bgr, (512, 512))

        tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0
        tensor = tensor.to(self.device, torch.float)

        hor_flipped = self._hor_flip(tensor)
        ver_flipped = self._ver_flip(tensor)

        with torch.no_grad():
            scores = [
                self._model(tensor),
                self._model(hor_flipped),
                self._model(ver_flipped),
                self._model(rotate(tensor, -5)),
                self._model(rotate(tensor, 5)),
                self._model(rotate(tensor, -8)),
                self._model(rotate(tensor, 8)),
                self._model(rotate(tensor, -12)),
                self._model(rotate(tensor, 12)),
                self._model(rotate(tensor, -15)),
                self._model(rotate(tensor, 15)),
            ]
            final_scores = sum(scores) / len(scores)
            probabilities = torch.softmax(final_scores, dim=1).squeeze().cpu().numpy()
            pred_class = torch.argmax(final_scores, dim=1).item()

        return ClassificationResult(
            pred_class=pred_class,
            class_name=CLASS_NAMES[pred_class],
            probabilities=probabilities.tolist(),
            class_names=CLASS_NAMES,
        )
