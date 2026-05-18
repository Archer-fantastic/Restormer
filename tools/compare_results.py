import os
import cv2
import numpy as np
from natsort import natsorted


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


def slice_images(images, slice_size=512):
    """对多个图像进行切片，边缘部分填充到相同尺寸"""
    # 获取图像尺寸
    h, w = images[0].shape[:2]
    
    # 存储所有切片
    all_slices = []
    
    # 滑窗切分，步长为slice_size
    patch_idx = 1
    for y_start in range(0, h, slice_size):
        for x_start in range(0, w, slice_size):
            y_end = min(y_start + slice_size, h)
            x_end = min(x_start + slice_size, w)
            
            # 计算实际切片尺寸
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            
            # 计算需要填充的尺寸（往左/往上填充）
            pad_left = 0
            pad_top = 0
            if actual_w < slice_size:
                pad_left = slice_size - actual_w
            if actual_h < slice_size:
                pad_top = slice_size - actual_h
            
            # 提取每个图像的切片并填充
            image_slices = []
            for img in images:
                slice_img = img[y_start:y_end, x_start:x_end]
                
                # 如果需要填充
                if pad_left > 0 or pad_top > 0:
                    # 创建填充后的图像（黑色填充）
                    padded_img = np.zeros((slice_size, slice_size, 3), dtype=np.uint8)
                    
                    # 将原切片放到右下角（往左/往上填充）
                    if pad_left > 0 and pad_top > 0:
                        padded_img[pad_top:, pad_left:, :] = slice_img
                    elif pad_left > 0:
                        padded_img[:, pad_left:, :] = slice_img
                    elif pad_top > 0:
                        padded_img[pad_top:, :, :] = slice_img
                    
                    slice_img = padded_img
                
                image_slices.append(slice_img)
            
            all_slices.append((image_slices, patch_idx))
            patch_idx += 1
    
    return all_slices


def create_grid(images, labels, max_cols=3, max_rows=3):
    """创建网格布局的图像，一行最多max_cols个，最多max_rows行"""
    # 限制图像数量
    images = images[:max_cols * max_rows]
    labels = labels[:max_cols * max_rows]
    
    # 计算网格尺寸
    num_images = len(images)
    num_cols = min(num_images, max_cols)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # 确保图像尺寸相同
    h, w = images[0].shape[:2]
    for img in images[1:]:
        if img.shape[:2] != (h, w):
            # 调整尺寸
            img = cv2.resize(img, (w, h))
    
    # 创建标签区域
    label_height = 50
    # 缝隙大小
    gap_size = 5
    
    # 创建网格图像
    grid_h = num_rows * (h + label_height) + (num_rows - 1) * gap_size
    grid_w = num_cols * w + (num_cols - 1) * gap_size
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # 添加图像和标签
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // num_cols
        col = i % num_cols
        
        # 计算位置（考虑缝隙）
        y_start = row * (h + label_height + gap_size)
        x_start = col * (w + gap_size)
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 0, 0)
        
        # 计算文本位置
        text_width = cv2.getTextSize(label, font, font_scale, font_thickness)[0][0]
        text_x = x_start + (w // 2) - (text_width // 2)
        text_y = y_start + (label_height // 2) + 10
        
        # 绘制文本
        cv2.putText(grid, label, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # 绘制分隔线
        cv2.line(grid, (x_start, y_start + label_height), (x_start + w, y_start + label_height), (0, 0, 0), 2)
        
        # 添加图像
        grid[y_start + label_height:y_start + label_height + h, x_start:x_start + w, :] = img
    
    return grid


def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Compare image restoration results')
    parser.add_argument('--folders', '-f', required=True, nargs='+', type=str, help='Folder paths to compare')
    parser.add_argument('--output', '-o', default='./results/comparison_results', type=str, help='Output folder path')
    parser.add_argument('--slice_size', '-s', default=512, type=int, help='Slice size')
    parser.add_argument('--max_cols', '-c', default=3, type=int, help='Maximum number of columns per row')
    parser.add_argument('--max_rows', '-r', default=3, type=int, help='Maximum number of rows')
    
    args = parser.parse_args()
    
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件夹中的图像文件
    folder_files = []
    for folder in args.folders:
        files = get_image_files_recursive(folder)
        file_map = {rel_path: full_path for full_path, rel_path in files}
        folder_files.append((folder, file_map))
    
    # 找到所有文件夹的共同文件
    if not folder_files:
        print("没有指定文件夹")
        return
    
    # 获取第一个文件夹的文件集合
    common_files = set(folder_files[0][1].keys())
    
    # 找出所有文件夹的共同文件
    for _, file_map in folder_files[1:]:
        common_files &= set(file_map.keys())
    
    print(f"找到 {len(common_files)} 个共同文件")
    
    # 处理每个共同文件
    for rel_path in natsorted(common_files):
        # 读取所有文件夹中的对应图像
        images = []
        folder_names = []
        
        for folder, file_map in folder_files:
            path = file_map[rel_path]
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"无法读取图像: {rel_path} 来自 {folder}")
                break
            images.append(img)
            folder_names.append(os.path.basename(folder))
        
        if len(images) != len(folder_files):
            continue
        
        # 检查所有图像尺寸是否相同
        ref_shape = images[0].shape
        for i in range(1, len(images)):
            if images[i].shape != ref_shape:
                # 调整尺寸
                images[i] = cv2.resize(images[i], (ref_shape[1], ref_shape[0]))
        
        # 生成切片
        slices = slice_images(images, args.slice_size)
        
        # 获取原始文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)
        
        # 处理每个切片
        for image_slices, patch_idx in slices:
            # 创建网格布局
            grid = create_grid(image_slices, folder_names, args.max_cols, args.max_rows)
            
            # 生成保存路径
            save_name = f"{base_name}_patch{patch_idx}.png"
            output_path = os.path.join(save_dir, save_name)
            
            # 保存图像
            success, encoded_img = cv2.imencode('.png', grid)
            if success:
                encoded_img.tofile(output_path)
                print(f"保存对比结果: {output_path}")
            else:
                print(f"保存失败: {output_path}")
    
    print("处理完成！")


if __name__ == "__main__":
    main()
