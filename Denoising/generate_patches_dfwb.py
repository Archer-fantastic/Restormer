import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

script_dir = r'Z:\14-调试数据\lxm\Dataset\DeNoise_Datasets'



src = os.path.join(script_dir, 'Downloads')
tar = os.path.join(script_dir, 'train', 'DFWB')
os.makedirs(tar, exist_ok=True)

patch_size = 512
overlap = 96
p_max = 800


def cv2_imread(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def cv2_imwrite(save_path, image_data):
    _, buf = cv2.imencode('.png', image_data)
    with open(save_path, 'wb') as f:
        f.write(buf)

def save_files(file_):
    path_contents = file_.split(os.sep)
    foldname = path_contents[-2]
    filename = os.path.splitext(path_contents[-1])[0]
    
    img = cv2_imread(file_)
    if img is None:
        return
        
    num_patch = 0
    w, h = img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size) 
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                patch = img[i:i+patch_size, j:j+patch_size,:]
                savename = os.path.join(tar, foldname + '-' + filename + '-' + str(num_patch) + '.png')
                cv2_imwrite(savename, patch)

    else:
        savename = os.path.join(tar, foldname + '-' + filename + '.png')
        cv2_imwrite(savename, img)


print(f"源目录: {src}")
print(f"目标目录: {tar}")

if not os.path.exists(src):
    print(f"错误: 源目录不存在: {src}")
    print("请先解压数据集文件!")
    input("按回车键退出...")
    exit(1)

files = []
for dataset in ['DIV2K', 'Flickr2K', 'WaterlooED', 'BSD400']:
    dataset_path = os.path.join(src, dataset)
    if os.path.exists(dataset_path):
        df = natsorted(glob(os.path.join(dataset_path, '*.png')) + glob(os.path.join(dataset_path, '*.jpg')) + glob(os.path.join(dataset_path, '*.bmp')))
        files.extend(df)
        print(f"找到 {dataset}: {len(df)} 张图像")

print(f"总计找到: {len(files)} 张图像")

if len(files) == 0:
    print("没有找到任何图像文件!")
    input("按回车键退出...")
    exit(1)

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print(f"使用 {num_cores} 个核心并行处理...")

Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))



print("完成!")
input("按回车键退出...")
