# Restormer 项目操作指南

## 一、项目概述

本项目是基于 Restormer 的图像处理工具集，支持图像去噪、去雨、运动去模糊等任务。核心模块位于 `basicsr/` 目录下，训练和测试脚本分别位于 `Denoising/`、`Deraining/`、`Motion_Deblurring/` 等子目录。

---

## 二、UI 界面操作说明

### 2.1 启动 UI

```bash
cd z:/14-调试数据/lxm/Projects/Restormer
python ui/denoise_gui.py
```

### 2.2 界面功能

| 区域 | 功能说明 |
|------|----------|
| **参数设置栏** | 模型类型、降噪模式、Sigma值、Tile大小 |
| **文件树** | 左侧显示文件夹目录结构 |
| **图像显示** | 左侧为原图，右侧为处理后效果图 |
| **控制台** | 显示处理日志和状态信息 |

### 2.3 操作流程

1. **选择文件夹**：点击 `Select Folder` 按钮，选择待处理图像所在目录
2. **选择模型**：
   - `denoise_blind`：盲降噪（单模型处理任意噪声水平）
   - `denoise_non_blind`：非盲降噪（固定Sigma值）
   - `derain`：去雨
   - `deblur`：运动去模糊
3. **设置参数**：
   - `Sigma`：噪声标准差（0-100）
   - `Tile Size`：分块处理大小（建议512-720）
4. **处理图像**：点击 `Process Image` 开始处理
5. **保存结果**：点击 `Save Image` 保存处理后的图像

---

## 三、绘制训练 PSNR 曲线图

### 3.1 使用方法

```bash
cd z:/14-调试数据/lxm/Projects/Restormer
python tools/实验结果绘图.py <日志目录路径>
```

### 3.2 功能说明

| 功能 | 说明 |
|------|------|
| 自动提取指标 | 从训练日志中提取 PSNR、L_pix 等指标 |
| 生成趋势图 | 输出 `{日志名}_psnr_chart.png` |
| 生成简要日志 | 输出 `{日志名}_brief.log`，包含训练参数汇总 |
| 批量处理 | 递归处理目录下所有 `.log` 文件 |

### 3.3 示例

```bash
# 绘制单个实验的PSNR曲线
python tools/实验结果绘图.py experiments/DCK/DCK_512_15/logs
```

---

## 四、Xray 图像切分（训练数据准备）

### 4.1 脚本说明

`Denoising/generate_patches_xray.py` 用于将大尺寸 Xray 图像切分为小 patches。

### 4.2 主要参数

```python
src = r'Z:\源图像目录'           # 原始大图像路径
tar = r'Z:\目标目录'             # 切分后保存路径
patch_size = 128                 # 每个patch的大小（可选：128/256/384/512/640/768）
overlap = 24 * (patch_size // 128)  # patch重叠区域
p_max = 200 * (patch_size // 128)   # 超过此尺寸才进行切分
```

### 4.3 修改与使用

```python
# 1. 修改源目录和目标目录
src = r'Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\HJJ\HJJ_原图数据\BMP原图\验证图像'
tar = r'Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\HJJ\HJJ_val_128'

# 2. 修改patch_size
patch_size = 128  # 根据训练配置选择合适的尺寸

# 3. 运行脚本
python Denoising/generate_patches_xray.py
```

### 4.4 注意事项

- 仅当图像宽高均超过 `p_max` 时才进行切分
- 小于阈值的图像直接保存原图
- 支持 `.png`、`.jpg`、`.bmp`、`.tif`、`.tiff` 格式

---

## 五、图像预测（推理）

### 5.1 脚本说明

`Denoising/test_gaussian_gray_denoising.py` 支持单张图片和批量文件夹处理。

### 5.2 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input_dir` | 输入图像目录 | 原始图像文件夹 |
| `--result_dir` | 结果保存目录 | `results/output` |
| `--weights` | 模型权重路径 | `experiments/DCK/models/net_g_best.pth` |
| `--config` | 配置文件路径 | `Denoising/Options/XRAY/DCK/DCK_512_15.py` |
| `--sigmas` | 噪声水平 | `15` |
| `--tile` | 瓦片大小 | `512` |
| `--model_type` | 模型类型 | `blind`（盲降噪）/ `non_blind` |
| `--recursive` | 递归处理子目录 | 默认启用 |
| `--preserve_structure` | 保留目录结构 | 默认启用 |

### 5.3 单张图片预测

```bash
python Denoising/test_gaussian_gray_denoising.py ^
    --input_dir "Z:\测试图像\单张.png" ^
    --result_dir "Z:\results\output" ^
    --weights "experiments\DCK\DCK_512_15\models\net_g_best.pth" ^
    --config "Denoising\Options\XRAY\DCK\DCK_512_15.py" ^
    --sigmas 15 ^
    --tile 512 ^
    --model_type blind ^
    --recursive False
```

### 5.4 批量文件夹预测

```bash
python Denoising/test_gaussian_gray_denoising.py ^
    --input_dir "Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\原始数据\DCK" ^
    --result_dir "Z:\14-调试数据\lxm\Projects\Restormer\results\01.电池壳_训练降噪效果" ^
    --weights "experiments\DCK\DCK_512_15\models\net_g_best.pth" ^
    --config "Denoising\Options\XRAY\DCK\DCK_512_15.py" ^
    --sigmas 15 ^
    --tile 512 ^
    --model_type blind
```

### 5.5 模型权重对应关系

| 模型类型 | 权重后缀 | 用途 |
|----------|----------|------|
| blind | `_blind.pth` | 盲降噪（任意噪声） |
| non_blind | `_sigma15.pth` | 固定Sigma=15 |
| non_blind | `_sigma25.pth` | 固定Sigma=25 |

---

## 六、启动训练

### 6.1 训练入口

`basicsr/train.py` 是统一的训练入口，通过配置文件驱动。

### 6.2 配置文件结构

项目提供三类配置文件：

| 目录 | 说明 | 示例 |
|------|------|------|
| `Denoising/Options/Origin/` | 原始官方配置 | `GaussianGrayDenoising_Restormer.yml` |
| `Denoising/Options/XRAY/` | Xray专项配置 | `XRAY/DCK/DCK_512_15.py` |
| `Denoising/Options/降噪/` | 自定义降噪配置 | `XRAY_DeNoise.py` |

### 6.3 配置文件关键参数

以 `Denoising/Options/XRAY/DCK/DCK_512_15.py` 为例：

```python
# 基本设置
NAME = "DCK_512_15"              # 实验名称
TOTAL_ITER = 50000               # 总迭代次数
SIGMA = 15                        # 噪声水平
GT_SIZE = 512                    # 训练patch大小

# 数据路径
Train_Dataroot_GT = r"Z:\数据集\训练集路径"
Val_Dataroot_GT = r"Z:\数据集\验证集路径"

# 网络结构
NETWORK_G = {
    "type": "Restormer",
    "inp_channels": 1,            # 灰度图用1，彩色图用3
    "out_channels": 1,
    "dim": 48,                    # 特征维度
    "num_blocks": [4, 6, 6, 8],  # 各阶段块数
}

# 训练参数
OPTIM_G = {
    "type": "AdamW",
    "lr": 1e-05,                  # 学习率
    "weight_decay": 0.0001,
}
```

### 6.4 启动训练命令

```bash
cd z:/14-调试数据/lxm/Projects/Restormer
python basicsr/train.py -opt Denoising/Options/XRAY/DCK/DCK_512_15.py
```

### 6.5 断点续训

训练会自动检测 `experiments/{实验名}/training_states/` 目录中的状态文件：
- `last.state`  优先使用
- `latest.state` 其次使用
- `best.state` 再次使用
- 数字状态文件 按编号选择最新

### 6.6 训练输出

| 输出类型 | 位置 |
|----------|------|
| 日志文件 | `experiments/{NAME}/logs/train_{NAME}_{时间戳}.log` |
| 模型权重 | `experiments/{NAME}/models/` |
| TensorBoard | `tb_logger/{NAME}/` |

---

## 七、常用工作流程

### 7.1 完整训练流程

```
1. 数据准备
   ├─ 原始图像 → generate_patches_xray.py → 切分patches
   └─ 训练集 + 验证集

2. 配置文件准备
   └─ 修改 Denoising/Options/XRAY/xxx.py

3. 启动训练
   └─ python basicsr/train.py -opt Denoising/Options/XRAY/xxx.py

4. 监控训练
   └─ python tools/实验结果绘图.py experiments/xxx/logs

5. 模型推理
   └─ python Denoising/test_gaussian_gray_denoising.py ...
```

### 7.2 快速推理流程

```
1. 准备权重文件 (net_g_best.pth)
2. 准备配置文件 (与训练配置匹配)
3. 执行推理脚本
4. 查看结果
```

---

## 八、注意事项

1. **GPU要求**：训练需要 CUDA 支持，建议至少 8GB 显存
2. **路径格式**：Windows 下建议使用 raw string (`r"path"`) 避免转义问题
3. **数据格式**：训练图像建议为 `.png` 格式，灰度图
4. **Tile大小**：处理大图像时建议设置 `--tile` 参数避免显存溢出
