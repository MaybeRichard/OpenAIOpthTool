import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from ..base import BaseModel, SegmentationResult
from ..registry import registry
from ..utils import load_image

REFUGE2_CLASS_NAMES = {
    0: "背景 (Background)",
    1: "视杯 (Optic Cup)",
    2: "视盘 (Optic Disc)",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@registry.register
class OpticDiscCupTransUNet(BaseModel):
    name = "optic_disc_cup_transunet"
    display_name = "视盘视杯分割 / Optic Disc & Cup Segmentation (TransUNet)"
    task = "segmentation"
    modality = "fundus"
    description = (
        "3-class segmentation (background/optic cup/optic disc) from fundus photos. "
        "TransUNet R50-ViT-B/16, trained on REFUGE2."
    )

    def __init__(self, checkpoint_path, device=None, num_classes=3, resolution=224):
        self.num_classes = num_classes
        self.resolution = resolution
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    def _load_model(self):
        # Add segmentation directory to sys.path for TransUNet imports
        seg_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "segmentation",
        )
        if seg_dir not in sys.path:
            sys.path.insert(0, seg_dir)

        from models.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
        from models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

        vit_config = CONFIGS_ViT_seg["R50-ViT-B_16"]
        grid_size = self.resolution // 16
        vit_config.patches.grid = (grid_size, grid_size)
        vit_config.n_classes = self.num_classes

        self._model = ViT_seg(
            vit_config, img_size=self.resolution, num_classes=self.num_classes
        )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            self._model.load_state_dict(ckpt["model_state_dict"])
        else:
            self._model.load_state_dict(ckpt)
        self._model.to(self.device)
        self._model.eval()

        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def predict(self, image) -> SegmentationResult:
        pil_image = load_image(image)
        original_size = (pil_image.height, pil_image.width)

        img_resized = TF.resize(
            pil_image,
            (self.resolution, self.resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = self._normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(img_tensor)

            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                mask = (probs > 0.5).long().squeeze(0).squeeze(0)
                probs_np = probs.squeeze(0).cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = probs.argmax(dim=1).squeeze(0)
                probs_np = probs.squeeze(0).cpu().numpy()

        mask_resized = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode="nearest",
            )
            .long()
            .squeeze()
            .cpu()
            .numpy()
        )

        probs_resized = (
            F.interpolate(
                torch.from_numpy(probs_np).unsqueeze(0),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )

        return SegmentationResult(
            mask=mask_resized,
            probabilities=probs_resized,
            num_classes=self.num_classes,
            class_names=REFUGE2_CLASS_NAMES,
            original_size=original_size,
        )
