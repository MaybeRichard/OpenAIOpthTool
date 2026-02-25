# -*- coding: utf-8 -*-
"""
病理性近视黄斑病变分类推理脚本

功能：传入眼底图像，输出病理性近视黄斑病变的分类结果（0-4类）
- 类别0: 正常 (Normal)
- 类别1: 豹纹状眼底 (Tessellated fundus)
- 类别2: 弥漫性脉络膜视网膜萎缩 (Diffuse chorioretinal atrophy)
- 类别3: 斑片状脉络膜视网膜萎缩 (Patchy chorioretinal atrophy)
- 类别4: 黄斑萎缩 (Macular atrophy)
"""

import os
import argparse
import cv2 as cv
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms.functional import rotate

join = os.path.join

# 类别名称映射
CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "豹纹状眼底 (Tessellated fundus)",
    2: "弥漫性脉络膜视网膜萎缩 (Diffuse chorioretinal atrophy)",
    3: "斑片状脉络膜视网膜萎缩 (Patchy chorioretinal atrophy)",
    4: "黄斑萎缩 (Macular atrophy)"
}


class ResNet18(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class InferenceModel:
    def __init__(self, checkpoint_path):
        """
        Args:
            checkpoint_path (str): 预训练权重文件的完整路径
        """
        self.checkpoint_path = checkpoint_path
        self.testHorFlip = transforms.RandomHorizontalFlip(p=1)
        self.testVerFlip = transforms.RandomVerticalFlip(p=1)

        # 设备选择：CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self):
        """加载模型和权重"""
        self.model = ResNet18(num_classes=5, pretrained=False)
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功，使用设备: {self.device}")

    def predict(self, image):
        """
        对单张图像进行预测（使用测试时增强TTA）

        Args:
            image (ndarray): 输入图像，shape为[H,W,C]，BGR格式（cv2.imread读取）

        Returns:
            pred_class (int): 预测的类别 (0-4)
            class_name (str): 类别名称
            probabilities (list): 各类别的概率
        """
        # 预处理：调整大小并归一化
        image = cv.resize(image, (512, 512))
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0
        image = image.to(self.device, torch.float)

        # 生成图像变体
        imageHorFlipped = self.testHorFlip(image)
        imageVerFlipped = self.testVerFlip(image)

        with torch.no_grad():
            # 原图 + 翻转 + 8种旋转 = 11个变体
            scores = [
                self.model(image),
                self.model(imageHorFlipped),
                self.model(imageVerFlipped),
                self.model(rotate(image, -5)),
                self.model(rotate(image, 5)),
                self.model(rotate(image, -8)),
                self.model(rotate(image, 8)),
                self.model(rotate(image, -12)),
                self.model(rotate(image, 12)),
                self.model(rotate(image, -15)),
                self.model(rotate(image, 15)),
            ]

            # 平均所有变体的预测结果
            final_scores = sum(scores) / len(scores)
            probabilities = torch.softmax(final_scores, dim=1).squeeze().cpu().numpy()

            _, pred_class = torch.max(final_scores, 1)
            pred_class = pred_class.item()

        return pred_class, CLASS_NAMES[pred_class], probabilities.tolist()


def main():
    parser = argparse.ArgumentParser(description="病理性近视黄斑病变分类推理")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--weights", "-w", type=str,
                        default=join(os.path.dirname(os.path.dirname(__file__)), "weights", "bestModel.pth"),
                        help="预训练权重路径 (默认: ../weights/bestModel.pth)")
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return

    if not os.path.exists(args.weights):
        print(f"错误: 权重文件不存在: {args.weights}")
        return

    # 加载模型
    model = InferenceModel(args.weights)

    # 读取图像
    image = cv.imread(args.image)
    if image is None:
        print(f"错误: 无法读取图像: {args.image}")
        return

    # 进行预测
    pred_class, class_name, probabilities = model.predict(image)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"输入图像: {args.image}")
    print("=" * 60)
    print(f"预测类别: {pred_class}")
    print(f"类别名称: {class_name}")
    print("-" * 60)
    print("各类别概率:")
    for i, prob in enumerate(probabilities):
        bar = "█" * int(prob * 30)
        print(f"  类别{i}: {prob:.4f} {bar} {CLASS_NAMES[i]}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# =============================================================================
# 运行命令示例:
#
# 1. 基本用法（使用默认权重路径）:
#    python code/inference.py --image /path/to/fundus_image.png
#
# 2. 指定权重路径:
#    python code/inference.py --image /path/to/fundus_image.png --weights ./weights/bestModel.pth
#
# 3. 简写形式:
#    python code/inference.py -i /path/to/fundus_image.png -w ./weights/bestModel.pth
# =============================================================================
