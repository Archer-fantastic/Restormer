# Restormer 项目 Conda 虚拟环境搭建指南
# 适用于 Windows (CUDA GPU 环境)
# 更新日期: 2026-04-23

# ==================== 快速开始 (3步) ====================
#
#   Step 1: 创建 conda 环境
#   Step 2: 安装 PyTorch (根据你的 CUDA 版本选择)
#   Step 3: 安装项目依赖
#
# ============================================================


# ==================== Step 1: 创建 conda 环境 ====================

conda create -n restormer python=3.10 -y
conda activate restormer


# ==================== Step 2: 安装 PyTorch (选一种) ====================
# --- 方案A: CUDA 11.x (推荐 RTX 20/30 系列) ---
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# --- 方案B: CUDA 12.x (推荐 RTX 40 系列) ---
# pip install torch==2.2.0+cu121 torchvision==0.17.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# --- 方案C: CPU only (无GPU时使用) ---
# pip install torch torchvision


# ==================== Step 3: 安装项目依赖 ====================

# 进入项目目录后执行:
cd z:/14-调试数据/lxm/Projects/Restormer
pip install -r requirements.txt


# ==================== 可选: 编译 CUDA 扩展 (训练需要) ====================
# 如果需要编译 basicsr 的 CUDA 扩展(如 DCN、FusedAct 等):
python setup.py develop
# 或跳过 CUDA 扩展:
# python setup.py develop --no_cuda_ext


# ==================== 验证安装 ====================

# 验证 Python 版本 (建议 >= 3.9, 推荐 3.10)
python --version

# 验证 PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 验证核心依赖
python -c "
import sys
deps = ['numpy', 'cv2', 'scipy', 'skimage', 'PIL', 'yaml', 'tqdm', 'natsort', 'matplotlib', 'einops', 'PyQt5']
for d in deps:
    try:
        m = __import__(d)
        print(f'[OK] {d}: {getattr(m,\"__version__\",\"OK\")}')
    except ImportError as e:
        print(f'[FAIL] {d}: {e}')
"

# 验证 Restormer 架构导入
python -c "from basicsr.models.archs.restormer_arch import Restormer; print('[OK] Restormer imported successfully')"


# ==================== 常见问题排查 ====================

# 问题1: PyQt5 安装失败 (Windows)
# 解决方案: 使用 conda 安装
# conda install pyqt -y

# 问题2: opencv-python 与 opencv-python-headless 冲突
# 解决方案: 二选一即可，优先 headless (服务器无GUI环境)
# pip uninstall opencv-python -y && pip install opencv-python-headless

# 问题3: CUDA 内存不足 (OOM)
# 解决方案: 减小 batch_size 或 tile_size，配置文件中修改:
#   BATCH_SIZE = 1
#   GT_SIZE = 256  # 从512降低到256

# 问题4: einops 找不到
# 解决方案:
# pip install einops --upgrade --force-reinstall

# 问题5: natsort 排序问题 (Windows中文路径)
# 解决方案: 确保版本 >= 8.0.0
# pip install natsort>=8.0.0 --upgrade


# ==================== 依赖用途说明 ====================
#
# | 依赖包          | 用途                          | 必需场景         |
# |----------------|-------------------------------|------------------|
# | torch          | 深度学习框架                 | 所有操作         |
# | torchvision    | 图像变换、数据增强            | 训练+推理       |
# | einops          | 张量重排(Restormer架构必需)  | **所有操作**     |
# | numpy           | 数值计算                     | 所有操作         |
# | opencv-python   | 图像读写/处理               | 所有操作         |
# | scikit-image    | 图像类型转换(img_as_ubyte)  | 推理+GUI        |
# | Pillow          | 图像格式支持                 | 数据加载         |
# | PyYAML          | 配置文件解析(.yml/.py)      | 训练+推理       |
# | scipy           | MATLAB .mat 文件读写        | SIDD/DND数据集   |
# | h5py            | HDF5 格式数据                | SIDD/DND数据集   |
# | lmdb            | 高速数据库                    | 大规模训练      |
# | tqdm            | 进度条                       | 推理             |
# | natsort          | 自然排序(支持中文路径)       | 推理+GUI        |
# | matplotlib      | 绘制PSNR曲线图               | 实验结果分析     |
# | PyQt5           | GUI界面                      | 仅GUI工具       |
# | wandb           | 远程实验跟踪                  | 可选             |
# | cython          | CUDA扩展编译                 | 编译安装        |
#
# ============================================================
