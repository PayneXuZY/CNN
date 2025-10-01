# -*- coding: utf-8 -*-
"""
用垃圾分类数据集中的单张图片做卷积直觉与实战演示
- 教学版：将图像缩到 5×5（灰度），用 3×3 全1卷积核做“滑窗”卷积，打印 5×5 输入与 3×3 输出
- 实战版：对原图(或设定尺寸)做 conv2d 与 max_pool2d，并可视化特征图
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

# ============ 配置区域 ============
# TODO: 把下面路径改成你数据集里任意一张图片
img_path = r"C:\Users\Payne\PycharmProjects\CIT64X\Data Modeling\Data Source\Plastic/plastic5.jpg"   # ←←← 改这里
resize_for_demo = (5, 5)     # 教学用的 5×5
resize_for_practice = (128, 128)  # 实战可视化用尺寸（可改为 224 等）
device = "cuda"               # 如有 GPU 并想用，可改 "cuda"

# ============ 工具函数 ============
def conv2d_single_channel(x, kernel, stride=1, padding=0, bias=0.0):
    """
    x: (H, W) 单通道输入
    kernel: (k, k)
    """
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode='constant')
    H, W = x.shape
    k = kernel.shape[0]
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    y = np.zeros((H_out, W_out), dtype=float)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+k, j*stride:j*stride+k]
            y[i, j] = np.sum(patch * kernel) + bias
    return y

def show_image_and_feature(original_img, feature_2d, title_img="Input", title_feat="Feature"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title(title_img)

    plt.subplot(1, 2, 2)
    plt.imshow(feature_2d, cmap='viridis')  # 不指定颜色也行，这里用默认方案
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title(title_feat)
    plt.tight_layout()
    plt.show()

# ============ 1) 加载图片 ============
p = Path(img_path)
if not p.exists():
    raise FileNotFoundError(f"找不到图片：{p}\n请把 img_path 改为你数据集里的一张图片路径。")

# 原图（用于展示）
img_rgb = Image.open(p).convert("RGB")

# ============ 2) 教学版：缩到 5×5，做“纸上滑窗”同款 ============
img_5x5 = img_rgb.convert("L").resize(resize_for_demo, Image.BILINEAR)  # 灰度 5×5
x_np = np.asarray(img_5x5, dtype=np.float32) / 255.0                     # 归一化到 [0,1]
kernel = np.ones((3, 3), dtype=np.float32)                               # 3×3 全1卷积核

y_valid = conv2d_single_channel(x_np, kernel, stride=1, padding=0, bias=0.0)
y_same  = conv2d_single_channel(x_np, kernel, stride=1, padding=1, bias=0.0)

print("=== 教学版：5×5 输入与 3×3 输出（valid）===")
print("输入 (5×5, 灰度, 归一化):\n", np.round(x_np, 3))
print("valid 卷积输出 (应为 3×3):\n", np.round(y_valid, 4))
print("same  卷积输出 (应为 5×5):\n", np.round(y_same, 4))

# 可视化（把 5×5 输入和 3×3 输出放一起看）
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(x_np, cmap='gray', vmin=0, vmax=1)
plt.title("5×5 灰度输入")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(y_valid, cmap='viridis')
plt.title("3×3 卷积输出（valid）")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')
plt.tight_layout()
plt.show()

# ============ 3) 实战版：用 PyTorch 对较大尺寸做 conv2d + pooling ============
to_tensor = transforms.Compose([
    transforms.Resize(resize_for_practice, interpolation=Image.BILINEAR),
    transforms.ToTensor(),  # → [C,H,W] 且归一化到 [0,1]
])

x_t = to_tensor(img_rgb).unsqueeze(0).to(device)  # [1,C,H,W]
N, C, H, W = x_t.shape
print(f"\n=== 实战版：原图经缩放后的形状: {x_t.shape} (NCHW) ===")

# 构造 3×3 平均卷积核（对每个输入通道分别做，再求和；这里先做单通道演示，再做 RGB）
# 单通道演示：把 RGB 转灰度后再做 conv
x_gray = transforms.functional.rgb_to_grayscale(img_rgb)
xg_t = transforms.ToTensor()(x_gray).unsqueeze(0).to(device)  # [1,1,H,W]
weight_1c = torch.ones((1, 1, 3, 3), device=device)  # Cout=1, Cin=1
bias_1c = torch.zeros(1, device=device)

y_valid_1c = F.conv2d(xg_t, weight_1c, bias=bias_1c, stride=1, padding=0)
y_same_1c  = F.conv2d(xg_t, weight_1c, bias=bias_1c, stride=1, padding=1)
pooled_1c  = F.max_pool2d(y_valid_1c, kernel_size=2, stride=2)

print("单通道 valid 输出形状:", tuple(y_valid_1c.shape))
print("单通道 same  输出形状:", tuple(y_same_1c.shape))
print("单通道 pooling 后形状:", tuple(pooled_1c.shape))

# 可视化：输入灰度图与单通道特征图
show_image_and_feature(x_gray, y_valid_1c.squeeze().cpu().numpy(),
                       title_img="灰度输入(缩放后)", title_feat="3×3 全1卷积（valid）特征图")

# 多通道演示（RGB）：权重 [Cout=1, Cin=3, k, k]，各通道权重都设为 1
weight_rgb = torch.ones((1, 3, 3, 3), device=device)
bias_rgb = torch.zeros(1, device=device)
y_rgb = F.conv2d(x_t, weight_rgb, bias=bias_rgb, stride=1, padding=0)  # [1,1,H-2,W-2]

print("RGB 多通道卷积后形状:", tuple(y_rgb.shape))
show_image_and_feature(img_rgb.resize(resize_for_practice), y_rgb[0, 0].cpu().numpy(),
                       title_img="RGB 输入(缩放后)", title_feat="RGB→1通道特征图（valid）")

# 你也可以将 kernel 改成边缘检测核、锐化核等，观察特征图差异：
# 例如：Sobel X 核（单通道）
sobel_x = torch.tensor([[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]], dtype=torch.float32, device=device).view(1,1,3,3)
sobel_feat = F.conv2d(xg_t, sobel_x, bias=None, stride=1, padding=1)  # same 风格
show_image_and_feature(x_gray, sobel_feat[0, 0].cpu().numpy(),
                       title_img="灰度输入(缩放后)", title_feat="Sobel X 边缘响应（same）")
