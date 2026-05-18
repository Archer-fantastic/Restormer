## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881



import numpy as np
import os
import argparse
import sys
from tqdm import tqdm

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default=r'results\01.电池壳降噪\原始图像', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=r'results\01.电池壳降噪\DeRain结果图像', type=str, help='Directory for results')
parser.add_argument('--weights', default='Deraining/pretrained_models/deraining.pth', type=str, help='Path to weights')
parser.add_argument('--recursive', action='store_true', default=True, help='Recursive search for images in subdirectories')
parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='Disable recursive search')
parser.add_argument('--preserve_structure', action='store_true', default=True, help='Preserve folder structure when saving')
parser.add_argument('--no-preserve_structure', action='store_false', dest='preserve_structure', help='Do not preserve folder structure')

args = parser.parse_args()


def get_image_files_recursive(input_dir):
    """递归查找所有图像文件，返回文件路径和对应的相对路径"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    files_with_relpath = []
    
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, input_dir)
                files_with_relpath.append((full_path, rel_path))
    
    files_with_relpath = natsorted(files_with_relpath, key=lambda x: x[1])
    return files_with_relpath


def get_image_files_flat(input_dir):
    """非递归查找图像文件"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    files = []
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(os.path.join(input_dir, filename))
    return natsorted(files), [os.path.basename(f) for f in files]

####### Load yaml #######
yaml_file = 'Deraining/Options/Deraining_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
# 直接处理输入目录下的所有图像
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

# 查找输入目录下的图像文件
if args.recursive:
    files_with_relpath = get_image_files_recursive(args.input_dir)
    files = [f[0] for f in files_with_relpath]
    rel_paths = [f[1] for f in files_with_relpath]
else:
    files, filenames = get_image_files_flat(args.input_dir)
    rel_paths = filenames

print(f"找到 {len(files)} 张图像")

with torch.no_grad():
    for idx, file_ in enumerate(tqdm(files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        # Inference
        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        # Save restored images
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        # 保存图像
        if args.preserve_structure:
            rel_path = rel_paths[idx]
            save_subdir = os.path.join(result_dir, os.path.dirname(rel_path))
            os.makedirs(save_subdir, exist_ok=True)
            save_file = os.path.join(save_subdir, os.path.basename(rel_path))
        else:
            save_file = os.path.join(result_dir, os.path.splitext(os.path.basename(file_))[0]+'.png')
        
        utils.save_img(save_file, restored)

print("处理完成！")
