import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# XRAY数据集路径
src = r'Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\HJJ\HJJ_原图数据\BMP原图\验证图像'
tar = r'Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\HJJ\HJJ_val_128'

os.makedirs(tar, exist_ok=True)


# 参数设置 - 针对XRAY大图像优化
# patch_size = 768  # 每个patch的大小

# 参数设置 - 针对XRAY大图像优化
# patch_size = 640  # 每个patch的大小

# 参数设置 - 针对XRAY大图像优化
# patch_size = 512  # 每个patch的大小

# 参数设置 - 针对XRAY大图像优化
# patch_size = 384  # 每个patch的大小

# 参数设置 - 针对XRAY大图像优化
# patch_size = 256  # 每个patch的大小
patch_size = 128  # 每个patch的大小

overlap = 24 * (patch_size // 128)       # patch之间的重叠区域
p_max = 200 * (patch_size // 128)       # 超过这个尺寸才进行切分


def cv2_imread(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  无法解码图像: {img_path}")
        return img
    except Exception as e:
        print(f"  读取失败: {e}")
        return None

def cv2_imwrite(save_path, image_data, ext):
    try:
        # 使用原始图像的扩展名进行保存
        _, buf = cv2.imencode(ext, image_data)
        with open(save_path, 'wb') as f:
            f.write(buf)
        return True
    except Exception as e:
        print(f"  保存失败: {e}")
        return False

def save_files(file_):
    path_contents = file_.split(os.sep)
    foldname = path_contents[-2]
    filename = os.path.splitext(path_contents[-1])[0]
    ext = os.path.splitext(path_contents[-1])[1].lower()
    
    img = cv2_imread(file_)
    if img is None:
        return
    
    # 转换为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 保持3通道格式
    
    num_patch = 0
    h, w = img.shape[:2]
    
    print(f"处理图像: {filename}, 尺寸: {w}x{h}, 格式: {ext}")
    
    if w > p_max and h > p_max:
        # 计算切分位置
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size) 
        h1.append(h-patch_size)
        
        print(f"  切分为 {len(w1)} x {len(h1)} = {len(w1)*len(h1)} 个patch")
        
        for i in w1:
            for j in h1:
                num_patch += 1
                # 检查索引是否越界
                if i + patch_size > w or j + patch_size > h:
                    print(f"  跳过越界patch: i={i}, j={j}, 尺寸: {w}x{h}")
                    continue
                # 注意：OpenCV图像索引是 (高度, 宽度, 通道)
                patch = img[j:j+patch_size, i:i+patch_size,:]
                if patch.size == 0:
                    print(f"  跳过空patch: i={i}, j={j}")
                    continue
                savename = os.path.join(tar, f"{foldname}_{filename}_p{num_patch}{ext}")
                cv2_imwrite(savename, patch, ext)
    else:
        # 小图像直接保存
        savename = os.path.join(tar, f"{foldname}_{filename}{ext}")
        cv2_imwrite(savename, img, ext)

print(f"源目录: {src}")
print(f"目标目录: {tar}")
print(f"Patch大小: {patch_size}, 重叠: {overlap}")

if not os.path.exists(src):
    print(f"错误: 源目录不存在: {src}")
    input("按回车键退出...")
    exit(1)

# 获取所有图像文件
files = glob(os.path.join(src, '*.png')) + glob(os.path.join(src, '*.jpg')) + glob(os.path.join(src, '*.bmp')) + glob(os.path.join(src, '*.tif')) + glob(os.path.join(src, '*.tiff'))
files = natsorted(files)

print(f"总计找到: {len(files)} 张图像")

if len(files) == 0:
    print("没有找到任何图像文件!")
    # input("按回车键退出...")
    # exit(1)

# 串行处理（更稳定）
for file_ in tqdm(files, desc="生成patches"):
    save_files(file_)

print(f"\n完成! 共生成patches保存到: {tar}")
