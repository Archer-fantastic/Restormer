#!/usr/bin/env python
import os
import shutil
import sys

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

data_dir = os.path.join(script_dir, 'Datasets')

zip_files = [
    'SIDD_train.zip',
    'SIDD_val.zip',
]

for zip_file in zip_files:
    zip_path = os.path.join(data_dir, zip_file)
    if os.path.exists(zip_path):
        print(f"解压: {zip_file}")
        try:
            shutil.unpack_archive(zip_path, data_dir)
            print(f"完成: {zip_file}")
        except Exception as e:
            print(f"解压 {zip_file} 失败: {e}")
    else:
        print(f"文件不存在: {zip_file}")

print("\n解压完成！")

for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        print(f"目录: {item}")

input("\n按回车键退出...")
