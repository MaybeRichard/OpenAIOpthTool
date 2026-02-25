# TransUNet Training Script for Medical Image Segmentation
# R50-ViT-B/16 with ImageNet-21k pretrained weights

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from datasets import SegmentationDataset, get_dataset_config
from losses import BCEDiceLoss, CEDiceLoss
from metrics import compute_dice_iou_binary, compute_dice_iou_multiclass, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description='TransUNet Medical Segmentation')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cvc', 'kvasir', 'refuge2'])
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--poly_power', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--pretrained_path', type=str,
                        default='pretrained/R50+ViT-B_16.npz')
    return parser.parse_args()


def poly_lr(optimizer, init_lr, epoch, max_epoch, power=0.9):
    """Polynomial learning rate decay."""
    lr = init_lr * (1 - epoch / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        if task == 'binary':
            loss = criterion(logits, masks)
            dice, iou = compute_dice_iou_binary(logits, masks)
        else:
            loss = criterion(logits, masks)
            dice, iou, _, _ = compute_dice_iou_multiclass(logits, masks,
                                                           num_classes=logits.shape[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        tracker.update(dice, iou, images.size(0))

    n = len(loader.dataset)
    return total_loss / n, tracker.avg_dice, tracker.avg_iou


@torch.no_grad()
def validate(model, loader, criterion, device, task, num_classes=1):
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        if task == 'binary':
            loss = criterion(logits, masks)
            dice, iou = compute_dice_iou_binary(logits, masks)
        else:
            loss = criterion(logits, masks)
            dice, iou, _, _ = compute_dice_iou_multiclass(logits, masks,
                                                           num_classes=num_classes)

        total_loss += loss.item() * images.size(0)
        tracker.update(dice, iou, images.size(0))

    n = len(loader.dataset)
    return total_loss / n, tracker.avg_dice, tracker.avg_iou


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')

    cfg = get_dataset_config(args.dataset)
    task = cfg['task']
    num_classes = cfg['num_classes']

    # For TransUNet: binary uses 2 classes (background + foreground) with CE
    # but we keep 1-class sigmoid for consistency with UNet
    # Actually TransUNet originally uses softmax with n_classes>=2
    # We adapt: binary tasks use 1 output + sigmoid, multi-class use n outputs + softmax

    # Datasets
    train_dataset = SegmentationDataset(args.dataset, split='train',
                                         resolution=args.resolution, seed=args.seed)
    val_dataset = SegmentationDataset(args.dataset, split='val',
                                       resolution=args.resolution, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Model: TransUNet R50-ViT-B_16
    vit_config = CONFIGS_ViT_seg[args.vit_name]

    # Adjust grid size for input resolution (grid = img_size // 16)
    grid_size = args.resolution // 16
    vit_config.patches.grid = (grid_size, grid_size)

    if task == 'binary':
        vit_config.n_classes = 1
        criterion = BCEDiceLoss()
    else:
        vit_config.n_classes = num_classes
        criterion = CEDiceLoss(num_classes=num_classes)

    model = ViT_seg(vit_config, img_size=args.resolution,
                    num_classes=vit_config.n_classes, zero_head=True)

    # Load pretrained weights
    if os.path.exists(args.pretrained_path):
        pretrained_weights = np.load(args.pretrained_path)
        model.load_from(pretrained_weights)
        print(f"Loaded pretrained weights from {args.pretrained_path}")
    else:
        print(f"WARNING: Pretrained weights not found at {args.pretrained_path}")

    model = model.to(device)

    # Optimizer: SGD with Polynomial LR decay
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    # Output directory
    save_dir = os.path.join(args.save_dir, f'transunet_{args.dataset}')
    os.makedirs(save_dir, exist_ok=True)

    best_dice = 0.0
    print(f"\n{'='*60}")
    print(f"TransUNet Training: {cfg['name']}")
    print(f"Task: {task}, Classes: {num_classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, BS: {args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Polynomial LR decay
        current_lr = poly_lr(optimizer, args.lr, epoch - 1, args.epochs,
                              power=args.poly_power)

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, task)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, task, num_classes)

        elapsed = time.time() - t0

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args),
            }, os.path.join(save_dir, 'best.pth'))

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
                  f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} | "
                  f"Best: {best_dice:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}s")

    # Save last
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'args': vars(args),
    }, os.path.join(save_dir, 'last.pth'))

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
