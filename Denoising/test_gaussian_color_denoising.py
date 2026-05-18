## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default=r'D:\Min\Projects\VSCodeProjects\03.DeNoise\dataset\RGB_Images', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=r'D:\Min\Projects\VSCodeProjects\03.DeNoise\05.Restormer-main\results\RGB_Images', type=str, help='Directory for results')
parser.add_argument('--weights', default=r'D:\Min\Projects\VSCodeProjects\03.DeNoise\05.Restormer-main\Denoising\pretrained_models\gaussian_color_denoising', type=str, help='Path to weights')
parser.add_argument('--model_type', default='non_blind', choices=['non_blind','blind'], type=str, help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')
parser.add_argument('--tile', type=int, default=512, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()

####### Load yaml #######
if args.model_type == 'blind':
    yaml_file = 'Options/原始配置文件/GaussianColorDenoising_Restormer.yml'
else:
    yaml_file = f'Options/原始配置文件/GaussianColorDenoising_RestormerSigma{args.sigmas.split(",")[0]}.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if not os.path.exists(yaml_file):
    print(f"Error: YAML file not found: {yaml_file}")
    # Try absolute path
    yaml_file = os.path.join(os.path.dirname(__file__), yaml_file)
    print(f"Trying: {yaml_file}")

x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = np.int_(args.sigmas.split(','))

factor = 8

datasets = ['']

for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration = Restormer(**x['network_g'])
    if args.model_type == 'blind':
        weights = args.weights+'_blind.pth'
    else:
        weights = args.weights + '_sigma' + str(sigma_test) +'.pth'
    checkpoint = torch.load(weights)
    model_restoration.load_state_dict(checkpoint['params'])

    print("===>Testing using weights: ",weights)
    print("------------------------------------------------")
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        # Recursive file search
        files = []
        for root, _, filenames in os.walk(inp_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.tif', '.jpg', '.jpeg', '.bmp')):
                    files.append(os.path.join(root, filename))
        files = natsorted(files)
        
        # files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        
        # result_dir_tmp = os.path.join(args.result_dir, args.model_type, dataset, str(sigma_test))
        # os.makedirs(result_dir_tmp, exist_ok=True)

        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img = np.float32(utils.load_img(file_))/255.

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
                    tile_overlap = args.tile_overlap

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

                # Preserve directory structure
                rel_path = os.path.relpath(file_, inp_dir)
                save_file = os.path.join(args.result_dir, args.model_type, dataset, str(sigma_test), rel_path)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                
                # save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                utils.save_img(save_file, img_as_ubyte(restored))
