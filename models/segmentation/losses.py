# Loss functions for medical image segmentation
# BCEDiceLoss for binary tasks, CEDiceLoss for multi-class tasks

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.

    Input: logits [B, 1, H, W] (before sigmoid)
    Target: binary mask [B, 1, H, W] in {0, 1}
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE
        bce_loss = self.bce(logits, targets)

        # Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CEDiceLoss(nn.Module):
    """Combined CE + Dice loss for multi-class segmentation.

    Input: logits [B, C, H, W] (before softmax)
    Target: class indices [B, H, W] in {0, ..., C-1}
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=3, smooth=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # CE loss
        ce_loss = self.ce(logits, targets)

        # Dice loss (per-class, then average)
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_onehot = F.one_hot(targets, self.num_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dice_sum = 0.0
        for c in range(self.num_classes):
            probs_c = probs[:, c].contiguous().view(probs.size(0), -1)
            targets_c = targets_onehot[:, c].contiguous().view(targets.size(0), -1)
            intersection = (probs_c * targets_c).sum(dim=1)
            dice_c = (2.0 * intersection + self.smooth) / (
                probs_c.sum(dim=1) + targets_c.sum(dim=1) + self.smooth
            )
            dice_sum += dice_c.mean()

        dice_loss = 1.0 - dice_sum / self.num_classes

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
