import sys
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# sys.stdout = open('debug_log.txt', 'w', encoding='utf-8')
# sys.stderr = sys.stdout

print("Starting script...")

import numpy as np
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

print("Imports 1 done")

from basicsr.models.archs.restormer_arch import Restormer

from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
# from pdb import set_trace as stx

print("Imports done")

parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising using Restormer')

parser.add_argument('--input_dir', default=r'Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\原始数据\DCK', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=r'Z:\14-调试数据\lxm\Projects\Restormer\results\01.电池壳_训练降噪效果', type=str, help='Directory for results')
parser.add_argument('--weights', default=r'experiments\DCK\DCK_512_15_5W\models\net_g_best.pth', type=str, help='Path to weights')
parser.add_argument('--config', default=r'Denoising\Options\XRAY\DCK\DCK_512_15.py', type=str, help='Path to configuration file (YAML or Python)')
parser.add_argument('--sigmas', default='15', type=str, help='Sigma values')
parser.add_argument('--tile', type=int, default=512, help='Tile size (e.g 720). None means testing on the original resolution image')
# parser.add_argument('--tile_overlap', type=int, default=48, help='Overlapping of different tiles')
parser.add_argument('--model_type', default='blind', choices=['non_blind','blind'], type=str, help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
parser.add_argument('--recursive', action='store_true', default=True, help='Recursive search for images in subdirectories (default: enabled)')
parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='Disable recursive search')
parser.add_argument('--preserve_structure', action='store_true', default=True, help='Preserve folder structure when saving (default: enabled)')
parser.add_argument('--no-preserve_structure', action='store_false', dest='preserve_structure', help='Do not preserve folder structure')

args = parser.parse_args()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


def get_image_files_recursive(input_dir):
    """递归查找所有图像文件，返回文件路径和对应的相对路径"""
    image_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.PNG', '.TIF', '.TIFF', '.JPG', '.JPEG']
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
    image_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.PNG', '.TIF', '.TIFF', '.JPG', '.JPEG']
    files = []
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(os.path.join(input_dir, filename))
    return natsorted(files), [os.path.basename(f) for f in files]

####### Load configuration #######
import yaml
import importlib.util

# 加载配置文件
def load_config(config_path):
    """加载配置文件，支持YAML和Python格式"""
    if config_path.endswith('.yml') or config_path.endswith('.yaml'):
        # 加载YAML文件
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(config_path, mode='r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=Loader)
        return config
    elif config_path.endswith('.py'):
        # 加载Python文件
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        # 转换为字典格式
        config = {
            'network_g': config_module.NETWORK_G
        }
        return config
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

# 确定配置文件路径
if args.config:
    # 使用用户提供的配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        exit(1)
    print(f"Loading config from: {config_path}")
    x = load_config(config_path)
else:
    # 使用默认的YAML文件
    if args.model_type == 'blind':
        yaml_file = 'Options/Origin/GaussianGrayDenoising_Restormer.yml'
    else:
        yaml_file = f'Options/Origin/GaussianGrayDenoising_RestormerSigma{args.sigmas.split(",")[0]}.yml'
    
    # 尝试相对路径
    if not os.path.exists(yaml_file):
        # 尝试绝对路径
        yaml_file = os.path.join(os.path.dirname(__file__), yaml_file)
        if not os.path.exists(yaml_file):
            print(f"Error: Default YAML file not found: {yaml_file}")
            exit(1)
    
    print(f"Loading default config from: {yaml_file}")
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = np.int_(args.sigmas.split(','))

factor = 8

datasets = ['']

for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration = Restormer(**x['network_g'])    
    # 检查用户提供的weights是否是完整的文件路径
    if args.weights.endswith('.pth'):
        # 如果是完整的文件路径，直接使用
        weights = args.weights
    else:
        # 否则，使用原来的拼接逻辑
        if args.model_type == 'blind':
            weights = args.weights+'_blind.pth'
        else:
            weights = args.weights + '_sigma' + str(sigma_test) +'.pth'
    
    if not os.path.exists(weights):
        print(f"Error: Weights file not found: {weights}")
        continue
        
    checkpoint = torch.load(weights)
    # 尝试不同的键名加载权重
    if 'params' in checkpoint:
        model_restoration.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model_restoration.load_state_dict(checkpoint['params_ema'])
    else:
        # 直接加载整个state_dict
        model_restoration.load_state_dict(checkpoint)

    print("===>Testing using weights: ",weights)
    print("------------------------------------------------")
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        
        if args.recursive:
            files_with_relpath = get_image_files_recursive(inp_dir)
            files = [f[0] for f in files_with_relpath]
            rel_paths = [f[1] for f in files_with_relpath]
        else:
            files, filenames = get_image_files_flat(inp_dir)
            rel_paths = filenames
        
        print(f"DEBUG: Input dir: {inp_dir}")
        print(f"DEBUG: Found {len(files)} files")
        result_dir_tmp = os.path.join(args.result_dir, f"{args.model_type}_{timestamp}", dataset, str(sigma_test))
        print(f"DEBUG: Creating result dir: {result_dir_tmp}")
        try:
            os.makedirs(result_dir_tmp, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create directory: {e}")
            continue

        with torch.no_grad():
            for idx, file_ in enumerate(tqdm(files)):
                try:
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    img = np.float32(utils.load_gray_img(file_))/255.

                    np.random.seed(seed=0)  # for reproducibility
                    img += np.random.normal(0, sigma_test/255., img.shape)

                    img = torch.from_numpy(img).permute(2,0,1)
                    input_ = img.unsqueeze(0).cuda()

                    # Padding in case images are not multiples of 8
                    h,w = input_.shape[2], input_.shape[3]
                    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                    padh = H-h if h%factor!=0 else 0
                    padw = W-w if w%factor!=0 else 0
                    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                    if args.tile is None:
                        restored = model_restoration(input_)
                    else:
                        # test the image tile by tile
                        b, c, h, w = input_.shape
                        tile = min(args.tile, h, w)
                        assert tile % 8 == 0, "tile size should be multiple of 8"
                        # tile_overlap = args.tile_overlap
                        tile_overlap = 24 * (tile // 128)

                        stride = tile - tile_overlap
                        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                        E = torch.zeros(b, c, h, w).type_as(input_)
                        W = torch.zeros_like(E)

                        for h_idx in h_idx_list:
                            for w_idx in w_idx_list:
                                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                                out_patch = model_restoration(in_patch)
                                out_patch_mask = torch.ones_like(out_patch)

                                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                        restored = E.div_(W)

                    # Unpad images to original dimensions
                    restored = restored[:,:,:h,:w]

                    restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                    if args.preserve_structure:
                        rel_path = rel_paths[idx]
                        save_subdir = os.path.join(result_dir_tmp, os.path.dirname(rel_path))
                        os.makedirs(save_subdir, exist_ok=True)
                        save_file = os.path.join(save_subdir, os.path.basename(rel_path))
                    else:
                        save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                    
                    utils.save_gray_img(save_file, img_as_ubyte(restored))
                    print(f"Saved: {save_file}")
                except Exception as e:
                    print(f"ERROR processing {file_}: {e}")
                    import traceback
                    traceback.print_exc()

print("Script finished.")
