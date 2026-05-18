#!/usr/bin/env python3
# 递归将所有YAML配置文件转换为Python配置文件

import os
import yaml
import glob

def escape_string(s):
    """转义字符串中的特殊字符"""
    return s.replace('\\', '\\\\').replace('\"', '\\"')

def format_value(value):
    """格式化值，为字符串添加引号"""
    if isinstance(value, str):
        return f'"{escape_string(value)}"'
    elif isinstance(value, dict):
        return format_dict(value)
    elif isinstance(value, list):
        return '[' + ', '.join(format_value(item) for item in value) + ']'
    else:
        return str(value)

def format_dict(d, indent=4):
    """格式化字典，使其更易读"""
    if not d:
        return '{}'
    
    lines = ['{']
    for key, value in d.items():
        if isinstance(value, dict):
            value_str = format_dict(value, indent + 4)
        elif isinstance(value, list):
            value_str = '[' + ', '.join(format_value(item) for item in value) + ']'
        else:
            value_str = format_value(value)
        lines.append(f'{" " * indent}"{key}": {value_str},')
    lines.append(' ' * (indent - 4) + '}')
    return '\n'.join(lines)

# 递归查找所有YAML文件
def find_yaml_files(directory):
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                yaml_files.append(os.path.join(root, file))
    return yaml_files

# 将YAML配置转换为Python配置文件
def convert_yml_to_py(yaml_file):
    # 读取YAML文件
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 生成Python文件名
    base_name = os.path.splitext(yaml_file)[0]
    py_file = f"{base_name}.py"
    
    # 构建Python配置内容
    file_basename = os.path.basename(yaml_file).replace('.yml', '').replace('.yaml', '')
    python_content = f"""# {file_basename} 配置模块
# 统一管理训练配置，实现"一改全改"

"""
    
    # 提取基本设置
    if 'name' in config:
        python_content += f"# 基本设置\nNAME = \"{escape_string(config['name'])}\"\n"
    if 'model_type' in config:
        python_content += f"MODEL_TYPE = \"{escape_string(config['model_type'])}\"\n"
    if 'scale' in config:
        python_content += f"SCALE = {config['scale']}\n"
    if 'num_gpu' in config:
        python_content += f"NUM_GPU = {config['num_gpu']}\n"
    if 'manual_seed' in config:
        python_content += f"MANUAL_SEED = {config['manual_seed']}\n\n"
    
    # 提取数据集设置
    if 'datasets' in config:
        python_content += f"# 数据集设置\nDATASETS = {format_dict(config['datasets'])}\n\n"
    
    # 提取网络结构
    if 'network_g' in config:
        python_content += f"# 网络结构\nNETWORK_G = {format_dict(config['network_g'])}\n\n"
    
    # 提取路径设置
    if 'path' in config:
        python_content += f"# 路径设置\nPATH = {format_dict(config['path'])}\n\n"
    
    # 提取训练设置
    if 'train' in config:
        train_config = config['train']
        python_content += "# 训练设置\n"
        if 'total_iter' in train_config:
            python_content += f"TOTAL_ITER = {train_config['total_iter']}  # 总迭代次数\n"
        if 'warmup_iter' in train_config:
            python_content += f"WARMUP_ITER = {train_config['warmup_iter']}\n"
        if 'use_grad_clip' in train_config:
            python_content += f"USE_GRAD_CLIP = {train_config['use_grad_clip']}\n"
        if 'use_amp' in train_config:
            python_content += f"USE_AMP = {train_config['use_amp']}\n"
        if 'amp_level' in train_config:
            python_content += f"AMP_LEVEL = \"{escape_string(train_config['amp_level'])}\"\n"
        
        # 提取调度器
        if 'scheduler' in train_config:
            python_content += f"\n# 学习率调度器\nSCHEDULER = {format_dict(train_config['scheduler'])}\n"
        
        # 提取数据增强
        if 'mixing_augs' in train_config:
            python_content += f"\n# 数据增强\nMIXING_AUGS = {format_dict(train_config['mixing_augs'])}\n"
        
        # 提取优化器
        if 'optim_g' in train_config:
            python_content += f"\n# 优化器\nOPTIM_G = {format_dict(train_config['optim_g'])}\n"
        
        # 提取损失函数
        if 'pixel_opt' in train_config:
            python_content += f"\n# 损失函数\nPIXEL_OPT = {format_dict(train_config['pixel_opt'])}\n\n"
    
    # 提取验证设置
    if 'val' in config:
        python_content += f"# 验证设置\nVAL = {format_dict(config['val'])}\n\n"
    
    # 提取日志设置
    if 'logger' in config:
        python_content += f"# 日志设置\nLOGGER = {format_dict(config['logger'])}\n\n"
    
    # 提取分布式训练设置
    if 'dist_params' in config:
        python_content += f"# 分布式训练设置\nDIST_PARAMS = {format_dict(config['dist_params'])}\n\n"
    
    # 添加分阶段训练迭代次数计算
    if 'datasets' in config and 'train' in config['datasets'] and 'gt_sizes' in config['datasets']['train']:
        python_content += "# 分阶段训练迭代次数 - 根据总迭代数自动计算\n"
        python_content += "STAGE_COUNT = len(DATASETS[\"train\"][\"gt_sizes\"])\n"
        python_content += "ITERS = [TOTAL_ITER // STAGE_COUNT] * STAGE_COUNT\n"
        python_content += "# 调整最后一个阶段的迭代次数，确保总和等于总迭代数\n"
        python_content += "ITERS[-1] += TOTAL_ITER - sum(ITERS)\n\n"
        python_content += "# 更新数据集配置中的迭代次数\n"
        python_content += "DATASETS[\"train\"][\"iters\"] = ITERS\n\n"
    
    # 添加配置信息打印函数
    python_content += "# 打印配置信息\ndef print_config():\n"
    python_content += "    print(\"=== 配置信息 ===\")\n"
    if 'name' in config:
        python_content += f"    print(f\"配置名称: {{NAME}}\")\n"
    if 'train' in config and 'total_iter' in config['train']:
        python_content += "    print(f\"总迭代数: {{TOTAL_ITER}}\")\n"
        if 'datasets' in config and 'train' in config['datasets'] and 'gt_sizes' in config['datasets']['train']:
            python_content += "    print(f\"分阶段迭代: {{ITERS}}\")\n"
        if 'scheduler' in config['train'] and 'periods' in config['train']['scheduler']:
            python_content += "    print(f\"调度器周期: {{SCHEDULER['periods']}}\")\n"
    python_content += "    print(\"====================\")\n\n"
    
    # 添加主函数
    python_content += "if __name__ == \"__main__\":\n"
    python_content += "    print_config()\n"
    
    # 保存Python配置文件
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(python_content)
    
    return py_file

def main():
    # 查找所有YAML文件
    yaml_files = find_yaml_files('Denoising/Options')
    print(f"找到 {len(yaml_files)} 个YAML配置文件")
    
    # 转换每个YAML文件
    for yaml_file in yaml_files:
        print(f"处理文件: {yaml_file}")
        try:
            py_file = convert_yml_to_py(yaml_file)
            print(f"  ✓ 已转换为: {os.path.basename(py_file)}")
        except Exception as e:
            print(f"  ✗ 转换失败: {e}")
    
    print("\n所有配置文件转换完成！")

if __name__ == "__main__":
    main()