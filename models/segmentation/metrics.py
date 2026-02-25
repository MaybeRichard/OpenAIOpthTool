# Evaluation metrics for medical image segmentation
# Dice coefficient and IoU (Intersection over Union)

import torch
import torch.nn.functional as F
import numpy as np


def compute_dice_iou_binary(pred_logits, targets, threshold=0.5):
    """Compute per-sample Dice and IoU for binary segmentation, then average.

    Args:
        pred_logits: [B, 1, H, W] logits (before sigmoid)
        targets: [B, 1, H, W] binary mask {0, 1}

    Returns:
        dice: scalar, mean per-sample foreground Dice
        iou: scalar, mean per-sample foreground IoU
    """
    B = pred_logits.size(0)
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()

    # Per-sample: flatten spatial dims only [B, N]
    preds_flat = preds.view(B, -1)
    targets_flat = targets.view(B, -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)  # [B]
    pred_sum = preds_flat.sum(dim=1)  # [B]
    target_sum = targets_flat.sum(dim=1)  # [B]

    smooth = 1e-6
    dice_per_sample = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)  # [B]
    iou_per_sample = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)  # [B]

    return dice_per_sample.mean().item(), iou_per_sample.mean().item()


def compute_dice_iou_multiclass(pred_logits, targets, num_classes=3):
    """Compute per-sample mean Dice and IoU for multi-class segmentation.

    For REFUGE2: report mean of optic cup (class 1) and optic disc (class 2).
    Computes Dice per sample per class, then averages.

    Args:
        pred_logits: [B, C, H, W] logits (before softmax)
        targets: [B, H, W] class indices {0, ..., C-1}

    Returns:
        mean_dice: mean per-sample Dice over foreground classes
        mean_iou: mean per-sample IoU over foreground classes
        per_class_dice: dict of {class_idx: mean_dice}
        per_class_iou: dict of {class_idx: mean_iou}
    """
    B = pred_logits.size(0)
    preds = pred_logits.argmax(dim=1)  # [B, H, W]
    smooth = 1e-6

    per_class_dice = {}
    per_class_iou = {}

    # Skip background (class 0), compute for foreground classes
    for c in range(1, num_classes):
        pred_c = (preds == c).float().view(B, -1)  # [B, N]
        target_c = (targets == c).float().view(B, -1)  # [B, N]

        intersection = (pred_c * target_c).sum(dim=1)  # [B]
        pred_sum = pred_c.sum(dim=1)  # [B]
        target_sum = target_c.sum(dim=1)  # [B]

        dice_per_sample = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou_per_sample = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)

        per_class_dice[c] = dice_per_sample.mean().item()
        per_class_iou[c] = iou_per_sample.mean().item()

    mean_dice = np.mean(list(per_class_dice.values()))
    mean_iou = np.mean(list(per_class_iou.values()))

    return mean_dice, mean_iou, per_class_dice, per_class_iou


class MetricTracker:
    """Track running averages of metrics during training/evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.count = 0

    def update(self, dice, iou, batch_size=1):
        self.dice_sum += dice * batch_size
        self.iou_sum += iou * batch_size
        self.count += batch_size

    @property
    def avg_dice(self):
        return self.dice_sum / max(self.count, 1)

    @property
    def avg_iou(self):
        return self.iou_sum / max(self.count, 1)
