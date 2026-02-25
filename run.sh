#!/bin/bash

# ===== Unified API Usage =====

# List all available models
python -c "
from opthtools import registry
for m in registry.list_models():
    print(f\"{m['name']:40s} {m['task']:15s} {m['modality']}\")
"

# Classification: Myopic Maculopathy Grading
# python -c "
# from opthtools import registry
# model = registry.load('myopic_maculopathy_resnet18', checkpoint_path='models/Classification/MyopicMaculopathyClassification/weights/bestModel.pth')
# result = model.predict('path/to/fundus_image.jpg')
# print(f'Class: {result.pred_class} - {result.class_name}')
# print(f'Probabilities: {result.probabilities}')
# "

# Segmentation: Optic Disc & Cup (UNet)
# python -c "
# from opthtools import registry
# model = registry.load('optic_disc_cup_unet', checkpoint_path='models/segmentation/segmentation_unet_refuge2/best.pth')
# result = model.predict('path/to/fundus_image.jpg')
# print(f'Mask shape: {result.mask.shape}')
# print(f'Classes: {result.class_names}')
# "

# ===== Raw Model Training Commands =====

# Train classification model
# python models/Classification/MyopicMaculopathyClassification/code/train.py --dataDir <data_dir> --weightDir <weight_dir>

# Train UNet segmentation
# python models/segmentation/train_unet.py --dataset refuge2 --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4

# Train TransUNet segmentation
# python models/segmentation/train_transunet.py --dataset refuge2 --gpu 0 --epochs 150 --batch_size 24 --lr 0.01

# Evaluate segmentation models
# python models/segmentation/evaluate.py --dataset all --model all --gpu 0
