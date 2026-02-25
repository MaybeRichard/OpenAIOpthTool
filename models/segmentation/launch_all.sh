#!/bin/bash
# Launch all 6 segmentation training jobs
cd /data/sichengli/Code/PixelGen/segmentation
PYTHON=/root/miniconda3/envs/python310/bin/python

# Kill any existing training processes
kill $(pgrep -f "train_unet.py") 2>/dev/null
kill $(pgrep -f "train_transunet.py") 2>/dev/null
sleep 2

# Clean up old checkpoints from smoke tests
rm -rf checkpoints/unet_* checkpoints/transunet_*

# Create logs directory
mkdir -p logs

echo "Launching 6 training jobs..."

# UNet Training (3 datasets on GPUs 0-2)
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON train_unet.py --dataset cvc --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_cvc.log 2>&1 &
echo "  [GPU 0] UNet CVC: PID $!"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON train_unet.py --dataset kvasir --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_kvasir.log 2>&1 &
echo "  [GPU 1] UNet Kvasir: PID $!"

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON train_unet.py --dataset refuge2 --gpu 0 --epochs 200 --batch_size 16 --lr 1e-4 > logs/unet_refuge2.log 2>&1 &
echo "  [GPU 2] UNet REFUGE2: PID $!"

# TransUNet Training (3 datasets on GPUs 3-5)
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON train_transunet.py --dataset cvc --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_cvc.log 2>&1 &
echo "  [GPU 3] TransUNet CVC: PID $!"

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON train_transunet.py --dataset kvasir --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_kvasir.log 2>&1 &
echo "  [GPU 4] TransUNet Kvasir: PID $!"

CUDA_VISIBLE_DEVICES=5 nohup $PYTHON train_transunet.py --dataset refuge2 --gpu 0 --epochs 150 --batch_size 24 --lr 0.01 > logs/transunet_refuge2.log 2>&1 &
echo "  [GPU 5] TransUNet REFUGE2: PID $!"

sleep 5
echo ""
echo "Running processes:"
ps aux | grep "train_" | grep python | grep -v grep
echo ""
echo "Total: $(ps aux | grep 'train_' | grep python | grep -v grep | wc -l) training processes"
