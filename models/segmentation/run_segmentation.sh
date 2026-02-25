#!/bin/bash
# Medical Image Segmentation Baselines - Run Commands
# Working directory: /data/sichengli/Code/PixelGen/segmentation

# ============================================================
# Step 0: Environment setup (run once)
# ============================================================
# pip install segmentation_models_pytorch ml_collections scipy

# Download TransUNet pretrained weights
# wget -P pretrained/ https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz

# ============================================================
# UNet Training (3 datasets on 3 GPUs)
# ============================================================

# UNet - CVC-ClinicDB (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python train_unet.py --dataset cvc --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_cvc.log 2>&1 &

# UNet - Kvasir-SEG (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python train_unet.py --dataset kvasir --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_kvasir.log 2>&1 &

# UNet - REFUGE2 (GPU 2)
CUDA_VISIBLE_DEVICES=2 nohup python train_unet.py --dataset refuge2 --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_refuge2.log 2>&1 &

# ============================================================
# TransUNet Training (3 datasets on 3 GPUs)
# ============================================================

# TransUNet - CVC-ClinicDB (GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup python train_transunet.py --dataset cvc --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_cvc.log 2>&1 &

# TransUNet - Kvasir-SEG (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup python train_transunet.py --dataset kvasir --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_kvasir.log 2>&1 &

# TransUNet - REFUGE2 (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup python train_transunet.py --dataset refuge2 --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_refuge2.log 2>&1 &

# ============================================================
# Evaluation (after training completes)
# ============================================================

# Evaluate all models on all datasets
# python evaluate.py --dataset all --model all --gpu 0
