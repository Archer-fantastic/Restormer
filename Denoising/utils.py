## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import cv2
import math

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# def load_img(filepath):
#     return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

# def save_img(filepath, img):
#     cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# def load_gray_img(filepath):
#     return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

# def save_gray_img(filepath, img):
#     cv2.imwrite(filepath, img)


def load_img(filepath):
    """加载彩色图像，支持中文路径"""
    # 使用 np.fromfile 读取二进制数据，再用 imdecode 解码
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    """保存彩色图像，支持中文路径"""
    # 转回 BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 编码为 jpg/png 等格式（默认 .png 如果扩展名不明确）
    success, encoded_img = cv2.imencode('.png', img_bgr)
    if not success:
        raise RuntimeError(f"图像编码失败: {filepath}")
    encoded_img.tofile(filepath)

def load_gray_img(filepath):
    """加载灰度图像，支持中文路径"""
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取灰度图像文件: {filepath}")
    return np.expand_dims(img, axis=2)  # (H, W) -> (H, W, 1)

def save_gray_img(filepath, img):
    """保存灰度图像，支持中文路径"""
    # 确保是二维灰度图 (H, W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)
    elif img.ndim != 2:
        raise ValueError("输入图像必须是灰度图 (H, W) 或 (H, W, 1)")
    
    success, encoded_img = cv2.imencode('.png', img)
    if not success:
        raise RuntimeError(f"灰度图像编码失败: {filepath}")
    encoded_img.tofile(filepath)