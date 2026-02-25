# Unified Evaluation Script for Medical Image Segmentation
# Loads best checkpoint and computes Dice + IoU on val set

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from models.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from datasets import SegmentationDataset, get_dataset_config
from metrics import compute_dice_iou_binary, compute_dice_iou_multiclass, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation Models')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cvc', 'kvasir', 'refuge2', 'all'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'transunet', 'all'])
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    return parser.parse_args()


def build_unet(task, num_classes):
    if task == 'binary':
        return smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=3, classes=1)
    else:
        return smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=3, classes=num_classes)


def build_transunet(task, num_classes, resolution):
    vit_config = CONFIGS_ViT_seg['R50-ViT-B_16']
    grid_size = resolution // 16
    vit_config.patches.grid = (grid_size, grid_size)
    if task == 'binary':
        vit_config.n_classes = 1
    else:
        vit_config.n_classes = num_classes
    return ViT_seg(vit_config, img_size=resolution,
                   num_classes=vit_config.n_classes)


@torch.no_grad()
def evaluate(model, loader, device, task, num_classes):
    model.eval()
    tracker = MetricTracker()
    all_per_class_dice = {}
    all_per_class_iou = {}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        if task == 'binary':
            dice, iou = compute_dice_iou_binary(logits, masks)
        else:
            dice, iou, pcd, pci = compute_dice_iou_multiclass(
                logits, masks, num_classes=num_classes)
            for c in pcd:
                all_per_class_dice.setdefault(c, []).append(pcd[c])
                all_per_class_iou.setdefault(c, []).append(pci[c])

        tracker.update(dice, iou, images.size(0))

    results = {
        'dice': tracker.avg_dice,
        'iou': tracker.avg_iou,
    }

    if task == 'multiclass':
        for c in all_per_class_dice:
            results[f'dice_class{c}'] = np.mean(all_per_class_dice[c])
            results[f'iou_class{c}'] = np.mean(all_per_class_iou[c])

    return results


def eval_one(dataset_name, model_name, args, device):
    cfg = get_dataset_config(dataset_name)
    task = cfg['task']
    num_classes = cfg['num_classes']

    # Dataset: REFUGE2 uses official test set, others use val split
    eval_split = 'test' if dataset_name == 'refuge2' else 'val'
    val_dataset = SegmentationDataset(dataset_name, split=eval_split,
                                       resolution=args.resolution, seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Model
    ckpt_path = os.path.join(args.save_dir, f'{model_name}_{dataset_name}', 'best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    if model_name == 'unet':
        model = build_unet(task, num_classes)
    else:
        model = build_transunet(task, num_classes, args.resolution)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    results = evaluate(model, val_loader, device, task, num_classes)
    results['epoch'] = ckpt.get('epoch', '?')
    return results


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    datasets = ['cvc', 'kvasir', 'refuge2'] if args.dataset == 'all' else [args.dataset]
    models = ['unet', 'transunet'] if args.model == 'all' else [args.model]

    print(f"\n{'='*70}")
    print(f"Medical Image Segmentation Evaluation")
    print(f"{'='*70}")

    all_results = []

    for ds in datasets:
        cfg = get_dataset_config(ds)
        for md in models:
            print(f"\n--- {cfg['name']} / {md.upper()} ---")
            results = eval_one(ds, md, args, device)
            if results is not None:
                all_results.append((ds, md, results))
                print(f"  Dice: {results['dice']:.4f}  IoU: {results['iou']:.4f}  "
                      f"(epoch {results['epoch']})")
                if cfg['task'] == 'multiclass':
                    for c in range(1, cfg['num_classes']):
                        dk = f'dice_class{c}'
                        ik = f'iou_class{c}'
                        if dk in results:
                            class_names = {1: 'Optic Cup', 2: 'Optic Disc'}
                            name = class_names.get(c, f'Class {c}')
                            print(f"    {name}: Dice={results[dk]:.4f}  IoU={results[ik]:.4f}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Dataset':<15} {'Model':<12} {'Dice':<10} {'IoU':<10}")
        print(f"{'-'*70}")
        for ds, md, res in all_results:
            cfg = get_dataset_config(ds)
            print(f"{cfg['name']:<15} {md.upper():<12} {res['dice']:<10.4f} {res['iou']:<10.4f}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
