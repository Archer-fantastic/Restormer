import os
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
 
def extract_metrics(log_file):
    """从日志文件中提取多个指标值"""
    metrics = {
        'psnr': [],
        'l_pix': [],
        'iterations': []
    }
    
    # 正则表达式匹配各指标值
    psnr_pattern = re.compile(r'psnr:\s*([0-9.]+)')
    l_pix_pattern = re.compile(r'l_pix:\s*([0-9.e-]+)')
    iter_pattern = re.compile(r'iter:\s*(\d+)')
    
    current_iter = 0
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 提取迭代次数
            iter_match = iter_pattern.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
            
            # 提取psnr值
            psnr_match = psnr_pattern.search(line)
            if psnr_match and current_iter > 0:
                psnr_value = float(psnr_match.group(1))
                metrics['psnr'].append(psnr_value)
                metrics['iterations'].append(current_iter)
            
            # 提取l_pix值
            l_pix_match = l_pix_pattern.search(line)
            if l_pix_match and current_iter > 0:
                l_pix_value = float(l_pix_match.group(1))
                metrics['l_pix'].append(l_pix_value)
    
    return metrics

def generate_chart(iterations, values, metric_name, output_file):
    """生成指标值的趋势图"""
    if not iterations or not values:
        print(f"没有找到{metric_name}值")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, values, 'b-', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel(f'{metric_name} Value')
    plt.title(f'{metric_name} Value Trend')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计信息
    if values:
        stats = f"Max: {max(values):.4f}\nMin: {min(values):.4f}\nAvg: {np.mean(values):.4f}"
        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"图表已保存到: {output_file}")

def extract_training_params(log_file):
    """从日志文件中提取训练和验证参数"""
    params = {
        'train_dataroot_gt': None,
        'train_dataroot_lq': None,
        'train_batch_size_per_gpu': None,
        'train_sigma_range': None,
        'val_dataroot_gt': None,
        'val_dataroot_lq': None,
        'val_sigma_test': None,
        'total_iter': None,
        'lr': None,
        'optimizer': None,
        'weight_decay': None,
        'gt_size': None,
        'dataset_enlarge_ratio': None,
        'val_freq': None,
        'save_freq': None,
        'model_type': None,
        'n_feat': None,
        'n_head': None,
        'n_layers': None
    }
    
    # 更精确的正则表达式，适应日志文件格式
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # 提取模型类型
        match = re.search(r'\s*model_type:\s*([a-zA-Z0-9_]+)', content)
        if match:
            params['model_type'] = match.group(1)
        
        # 提取训练参数
        match = re.search(r'\s*total_iter:\s*(\d+)', content)
        if match:
            params['total_iter'] = match.group(1)
        
        match = re.search(r'\s*lr:\s*([0-9.e-]+)', content)
        if match:
            params['lr'] = match.group(1)
        
        match = re.search(r'\s*weight_decay:\s*([0-9.e-]+)', content)
        if match:
            params['weight_decay'] = match.group(1)
        
        match = re.search(r'\s*val_freq:\s*(\d+)', content)
        if match:
            params['val_freq'] = match.group(1)
        
        match = re.search(r'\s*save_checkpoint_freq:\s*(\d+)', content)
        if match:
            params['save_freq'] = match.group(1)
        
        # 提取训练数据集参数
        match = re.search(r'train:\[[\s\S]*?dataroot_gt:\s*([^\n]+)', content)
        if match:
            params['train_dataroot_gt'] = match.group(1).strip()
        
        match = re.search(r'train:\[[\s\S]*?dataroot_lq:\s*([^\n]+)', content)
        if match:
            params['train_dataroot_lq'] = match.group(1).strip()
        
        match = re.search(r'train:\[[\s\S]*?batch_size_per_gpu:\s*(\d+)', content)
        if match:
            params['train_batch_size_per_gpu'] = match.group(1)
        
        match = re.search(r'train:\[[\s\S]*?sigma_range:\s*(\d+)', content)
        if match:
            params['train_sigma_range'] = match.group(1)
        
        match = re.search(r'train:\[[\s\S]*?gt_size:\s*(\d+)', content)
        if match:
            params['gt_size'] = match.group(1)
        
        match = re.search(r'train:\[[\s\S]*?dataset_enlarge_ratio:\s*(\d+)', content)
        if match:
            params['dataset_enlarge_ratio'] = match.group(1)
        
        # 提取验证数据集参数
        match = re.search(r'val:\[[\s\S]*?dataroot_gt:\s*([^\n]+)', content)
        if match:
            params['val_dataroot_gt'] = match.group(1).strip()
        
        match = re.search(r'val:\[[\s\S]*?dataroot_lq:\s*([^\n]+)', content)
        if match:
            params['val_dataroot_lq'] = match.group(1).strip()
        
        match = re.search(r'val:\[[\s\S]*?sigma_test:\s*(\d+)', content)
        if match:
            params['val_sigma_test'] = match.group(1)
        
        # 提取模型参数
        match = re.search(r'network_g:\[[\s\S]*?dim:\s*(\d+)', content)
        if match:
            params['n_feat'] = match.group(1)
        
        match = re.search(r'network_g:\[[\s\S]*?heads:\s*\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', content)
        if match:
            params['n_head'] = match.group(1)
        
        match = re.search(r'network_g:\[[\s\S]*?num_blocks:\s*\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', content)
        if match:
            params['n_layers'] = match.group(1)
        
        # 提取优化器信息
        match = re.search(r'optim_g:\[[\s\S]*?type:\s*([a-zA-Z0-9_]+)', content)
        if match:
            params['optimizer'] = match.group(1)
    
    return params

def generate_brief_log(params, log_file):
    """生成简要日志文件"""
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    # 保存到与日志文件同一目录
    output_file = os.path.join(os.path.dirname(log_file), f"{base_name}_brief.log")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Training and Validation Parameters\n")
        f.write("====================================\n\n")
        f.write("Training Data Parameters:\n")
        f.write(f"  dataroot_gt: {params['train_dataroot_gt']}\n")
        f.write(f"  dataroot_lq: {params['train_dataroot_lq']}\n")
        f.write(f"  batch_size_per_gpu: {params['train_batch_size_per_gpu']}\n")
        f.write(f"  sigma_range: {params['train_sigma_range']}\n")
        f.write(f"  gt_size: {params['gt_size']}\n")
        f.write(f"  dataset_enlarge_ratio: {params['dataset_enlarge_ratio']}\n\n")
        
        f.write("Validation Data Parameters:\n")
        f.write(f"  dataroot_gt: {params['val_dataroot_gt']}\n")
        f.write(f"  dataroot_lq: {params['val_dataroot_lq']}\n")
        f.write(f"  sigma_test: {params['val_sigma_test']}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  total_iter: {params['total_iter']}\n")
        f.write(f"  lr: {params['lr']}\n")
        f.write(f"  optimizer: {params['optimizer']}\n")
        f.write(f"  weight_decay: {params['weight_decay']}\n")
        f.write(f"  val_freq: {params['val_freq']}\n")
        f.write(f"  save_freq: {params['save_freq']}\n\n")
        
        f.write("Model Parameters:\n")
        f.write(f"  model_type: {params['model_type']}\n")
        f.write(f"  n_feat: {params['n_feat']}\n")
        f.write(f"  n_head: {params['n_head']}\n")
        f.write(f"  n_layers: {params['n_layers']}\n")
    
    print(f"简要日志已保存到: {output_file}")

def process_directory(directory):
    """递归处理目录中的所有日志文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            # 跳过_brief.log文件
            if file.endswith('.log') and not file.endswith('_brief.log'):
                log_file = os.path.join(root, file)
                print(f"处理文件: {log_file}")
                
                # 提取训练参数并生成简要日志
                params = extract_training_params(log_file)
                generate_brief_log(params, log_file)
                
                # 提取指标值
                metrics = extract_metrics(log_file)
                
                # 生成psnr图表
                if metrics['psnr']:
                    base_name = os.path.splitext(file)[0]
                    # 保存到与日志文件同一目录
                    output_file = os.path.join(os.path.dirname(log_file), f"{base_name}_psnr_chart.png")
                    # 确保iterations和psnr长度匹配
                    psnr_iterations = metrics['iterations'][:len(metrics['psnr'])]
                    generate_chart(psnr_iterations, metrics['psnr'], 'PSNR', output_file)
                else:
                    print(f"文件 {log_file} 中没有找到psnr值")
                
                # 生成l_pix图表
                # if metrics['l_pix']:
                #     base_name = os.path.splitext(file)[0]
                #     # 保存到与日志文件同一目录
                #     output_file = os.path.join(os.path.dirname(log_file), f"{base_name}_l_pix_chart.png")
                #     # l_pix的迭代次数需要单独提取，因为它和psnr的迭代次数可能不同
                #     # 重新提取l_pix的迭代次数
                #     l_pix_iterations = []
                #     current_iter = 0
                #     with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                #         for line in f:
                #             iter_match = re.search(r'iter:\s*(\d+)', line)
                #             if iter_match:
                #                 current_iter = int(iter_match.group(1))
                #             l_pix_match = re.search(r'l_pix:\s*([0-9.e-]+)', line)
                #             if l_pix_match and current_iter > 0:
                #                 l_pix_iterations.append(current_iter)
                #     # 确保长度匹配
                #     l_pix_iterations = l_pix_iterations[:len(metrics['l_pix'])]
                #     generate_chart(l_pix_iterations, metrics['l_pix'], 'L_PIX', output_file)
                # else:
                #     print(f"文件 {log_file} 中没有找到l_pix值")

def main():
    parser = argparse.ArgumentParser(description='统计日志文件中的指标值并生成趋势图')
    parser.add_argument('folder', help='要处理的目录路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"目录不存在: {args.folder}")
        return
    
    process_directory(args.folder)

if __name__ == '__main__':
    main()
