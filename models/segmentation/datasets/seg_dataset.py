# Unified Segmentation Dataset for CVC-ClinicDB, Kvasir-SEG, REFUGE2
# Split logic matches PixelGen generation experiments exactly

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# Dataset configurations
DATASET_CONFIGS = {
    'cvc': {
        'name': 'CVC-ClinicDB',
        'data_root': '/data2/sichengli/Data/test/Segmentation/CVC-ClinicDB',
        'img_subdir': 'PNG/Original',
        'mask_subdir': 'PNG/Ground Truth',
        'img_ext': ['.png'],
        'num_classes': 1,  # binary
        'task': 'binary',
    },
    'kvasir': {
        'name': 'Kvasir-SEG',
        'data_root': '/data2/sichengli/Data/test/Segmentation/Kvasir-SEG/Kvasir-SEG',
        'img_subdir': 'images',
        'mask_subdir': 'masks',
        'img_ext': ['.jpg', '.png', '.jpeg'],
        'num_classes': 1,  # binary
        'task': 'binary',
    },
    'refuge2': {
        'name': 'REFUGE2',
        'data_root': '/data2/sichengli/Data/test/Segmentation/REFUGE2',
        'splits': ['train', 'val'],  # combine train+val for training pool
        'img_subdir': 'images',
        'mask_subdir': 'mask',
        'img_ext': ['.jpg', '.png', '.jpeg'],
        'mask_ext': ['.bmp', '.png', '.jpg'],
        'num_classes': 3,  # background, optic cup, optic disc
        'task': 'multiclass',
        # pixel values: 0=background, 128=optic cup, 255=optic disc
        'class_mapping': {0: 0, 128: 1, 255: 2},
    },
}


def get_dataset_config(dataset_name):
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


class SegmentationDataset(Dataset):
    """
    Unified segmentation dataset for medical image segmentation baselines.

    Supports CVC-ClinicDB, Kvasir-SEG, and REFUGE2 with split logic
    matching the PixelGen generation experiments exactly (seed=42, 90/10 split).

    For binary tasks (CVC, Kvasir): mask is {0, 1} single channel
    For multi-class (REFUGE2): mask is class index map {0, 1, 2}

    Images are normalized with ImageNet mean/std (pretrained backbone standard).
    """

    # ImageNet normalization for pretrained backbones
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, dataset_name, split='train', resolution=224,
                 train_ratio=0.9, val_ratio=0.1, seed=42, augment=True):
        super().__init__()
        self.config = get_dataset_config(dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.resolution = resolution
        self.augment = augment and (split == 'train')

        # Collect image-mask pairs
        if dataset_name == 'refuge2':
            self.pairs = self._collect_refuge2(split, train_ratio, val_ratio, seed)
        else:
            self.pairs = self._collect_simple(split, train_ratio, seed)

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
        )

        print(f"[SegmentationDataset-{self.config['name']}] {split}: {len(self.pairs)} samples")

    def _collect_simple(self, split, train_ratio, seed):
        """Collect pairs for CVC-ClinicDB and Kvasir-SEG.
        Exactly matches PixelGen split logic."""
        cfg = self.config
        img_dir = os.path.join(cfg['data_root'], cfg['img_subdir'])
        mask_dir = os.path.join(cfg['data_root'], cfg['mask_subdir'])

        all_files = sorted([
            f for f in os.listdir(img_dir)
            if any(f.endswith(ext) for ext in cfg['img_ext'])
        ])

        # Split by index - matches PixelGen exactly
        random.seed(seed)
        indices = list(range(len(all_files)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)

        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        selected_files = [all_files[i] for i in sorted(selected_indices)]

        pairs = []
        for f in selected_files:
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, f)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))

        return pairs

    def _collect_refuge2_split(self, official_splits):
        """Collect image-mask pairs from specified REFUGE2 official splits."""
        cfg = self.config
        pairs = []
        for s in official_splits:
            img_dir = os.path.join(cfg['data_root'], s, cfg['img_subdir'])
            mask_dir = os.path.join(cfg['data_root'], s, cfg['mask_subdir'])

            img_files = sorted([
                f for f in os.listdir(img_dir)
                if any(f.endswith(ext) for ext in cfg['img_ext'])
            ])

            for img_f in img_files:
                img_path = os.path.join(img_dir, img_f)
                base_name = os.path.splitext(img_f)[0]
                mask_path = None
                for ext in cfg['mask_ext']:
                    candidate = os.path.join(mask_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                if mask_path is not None:
                    pairs.append((img_path, mask_path))
        return pairs

    def _collect_refuge2(self, split, train_ratio, val_ratio, seed):
        """Collect pairs for REFUGE2.
        train: official train+val combined, with 10% random holdout for val monitoring.
        val: holdout from train+val (for training-time monitoring).
        test: official test set (for final evaluation)."""
        if split == 'test':
            # Official test set
            return self._collect_refuge2_split(['test'])

        # train/val: combine official train+val, then holdout
        all_pairs = self._collect_refuge2_split(['train', 'val'])

        random.seed(seed)
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * (1 - val_ratio))

        if split == 'train':
            return all_pairs[:split_idx]
        else:  # val
            return all_pairs[split_idx:]

    def _process_mask(self, mask_pil):
        """Convert PIL mask to tensor based on task type."""
        mask_np = np.array(mask_pil)

        if self.config['task'] == 'binary':
            # Threshold to binary {0, 1}
            mask_np = (mask_np > 127).astype(np.float32)
            return torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]
        else:
            # Multi-class: map pixel values to class indices
            class_map = self.config['class_mapping']
            result = np.zeros_like(mask_np, dtype=np.int64)
            for pixel_val, class_idx in class_map.items():
                result[mask_np == pixel_val] = class_idx
            # Handle values not in mapping (due to interpolation artifacts)
            # Map to nearest defined class
            for pixel_val in np.unique(mask_np):
                if pixel_val not in class_map:
                    closest = min(class_map.keys(), key=lambda x: abs(x - pixel_val))
                    result[mask_np == pixel_val] = class_map[closest]
            return torch.from_numpy(result).long()  # [H, W]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize
        image = TF.resize(image, (self.resolution, self.resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.resolution, self.resolution),
                         interpolation=transforms.InterpolationMode.NEAREST)

        # Augmentation (train only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation (0/90/180/270)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Color jitter (image only)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
                image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))

        # Image to tensor and normalize with ImageNet stats
        image_tensor = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        image_tensor = self.normalize(image_tensor)

        # Process mask
        mask_tensor = self._process_mask(mask)

        return image_tensor, mask_tensor
