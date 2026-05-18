# HF_256 配置模块
# 统一管理训练配置，实现"一改全改"

# 基本设置
NAME = "DCK_256_25"
TOTAL_ITER = 50000  # 总迭代次数 - 修改这里可以自动更新其他相关参数
SIGMA = 25
Train_Dataroot_GT = r"Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\电池壳\DCK_train_256"
Train_Dataroot_LQ = "none"
Val_Dataroot_GT = r"Z:\14-调试数据\lxm\Dataset\DeNoise_XRAY\电池壳\DCK_val_256"
Val_Dataroot_LQ = "none"

Pretrain_Network_G = r"./Denoising/pretrained_models/gaussian_gray_denoising_blind.pth"
WARMUP_ITER = 500

GT_SIZE = 256  # 修改这里可以自动更新gt_sizes和mini_batch_sizes
BATCH_SIZE = 1  # 每个GPU的批次大小
MODEL_TYPE = "ImageCleanModel"
SCALE = 1
NUM_GPU = 1
MANUAL_SEED = 100

# 自动生成gt_sizes和mini_batch_sizes
def generate_gt_sizes(gt_size):
    """根据GT_SIZE自动生成gt_sizes列表"""
    sizes = []
    current = 128
    while current <= gt_size:
        sizes.append(current)
        current += 128
    return sizes

def generate_mini_batch_sizes(gt_sizes):
    """根据gt_sizes自动生成mini_batch_sizes列表"""
    batch_sizes = []
    for size in gt_sizes:
        if size == 128:
            batch_sizes.append(2)
        elif size == 256:
            batch_sizes.append(2)
        elif size == 384:
            batch_sizes.append(1)
        elif size == 512:
            batch_sizes.append(1)
        else:
            batch_sizes.append(1)
    return batch_sizes

# 生成gt_sizes和mini_batch_sizes
GT_SIZES = generate_gt_sizes(GT_SIZE)
MINI_BATCH_SIZES = generate_mini_batch_sizes(GT_SIZES)

# 数据集设置
DATASETS = {
    "train": {
        "name": "TrainSet",
        "type": "Dataset_GaussianDenoising",
        "sigma_type": "constant",
        "sigma_range": SIGMA,
        "in_ch": 1,
        "dataroot_gt": Train_Dataroot_GT,
        "dataroot_lq": Train_Dataroot_LQ,
        "geometric_augs": True,
        "filename_tmpl": "{}",
        "io_backend": {
            "type": "disk",
        },
        "use_shuffle": True,
        "num_worker_per_gpu": 2,
        "batch_size_per_gpu": BATCH_SIZE,
        "gt_size": GT_SIZE,
        "gt_sizes": GT_SIZES,
        "mini_batch_sizes": MINI_BATCH_SIZES,
        "iters": [],  # 会被自动计算并更新
        "dataset_enlarge_ratio": 10,
        "prefetch_mode": "cuda",
        "pin_memory": True,
    },
    "val": {
        "name": "ValSet",
        "type": "Dataset_GaussianDenoising",
        "sigma_test": SIGMA,
        "in_ch": 1,
        "dataroot_gt": Val_Dataroot_GT,
        "dataroot_lq": Val_Dataroot_LQ,
        "io_backend": {
            "type": "disk",
        },
    },
}

# 网络结构
NETWORK_G = {
    "type": "Restormer",
    "inp_channels": 1,
    "out_channels": 1,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "BiasFree",
    "dual_pixel_task": False,
}

# 路径设置
PATH = {
    "pretrain_network_g": Pretrain_Network_G,
    "strict_load_g": True,
    "resume_state": None,
}

# 训练设置
USE_GRAD_CLIP = True
USE_AMP = True
AMP_LEVEL = "O1"

# 学习率调度器
# 自动计算调度器周期，基于总迭代数
PERIOD1 = TOTAL_ITER // 3
PERIOD2 = TOTAL_ITER - PERIOD1

SCHEDULER = {
    "type": "CosineAnnealingRestartCyclicLR",
    "periods": [PERIOD1, PERIOD2],
    "restart_weights": [1, 1],
    "eta_mins": [1e-05, 1e-06],
}

# 数据增强
MIXING_AUGS = {
    "mixup": True,
    "mixup_beta": 1.2,
    "use_identity": True,
}

# 优化器
OPTIM_G = {
    "type": "AdamW",
    "lr": 1e-05,
    "weight_decay": 0.0001,
    "betas": [0.9, 0.999],
}

# 损失函数
PIXEL_OPT = {
    "type": "L1Loss",
    "loss_weight": 1,
    "reduction": "mean",
}

# 验证设置
VAL = {
    "window_size": 8,
    "val_freq": 1000.0,
    "save_img": False,
    "rgb2bgr": True,
    "use_image": False,
    "max_minibatch": 2,
    "metrics": {
        "psnr": {
            "type": "calculate_psnr",
            "crop_border": 0,
            "test_y_channel": False,
        },
    },
}

# 日志设置
LOGGER = {
    "print_freq": 100,
    "save_checkpoint_freq": 0,  # 0表示不按频率保存
    "save_best": True,  # 保存最佳模型
    "save_last": True,  # 保存最后一个模型
    "use_tb_logger": False,
    "wandb": {
        "project": None,
        "resume_id": None,
    },
}

# 分布式训练设置
DIST_PARAMS = {
    "backend": "nccl",
    "port": 29500,
}

# 分阶段训练迭代次数 - 根据总迭代数自动计算
STAGE_COUNT = len(DATASETS["train"]["gt_sizes"])
ITERS = [TOTAL_ITER // STAGE_COUNT] * STAGE_COUNT
# 调整最后一个阶段的迭代次数，确保总和等于总迭代数
ITERS[-1] += TOTAL_ITER - sum(ITERS)

# 更新数据集配置中的迭代次数
DATASETS["train"]["iters"] = ITERS

# 打印配置信息
def print_config():
    print("=== 配置信息 ===")
    print(f"配置名称: {NAME}")
    print(f"总迭代数: {TOTAL_ITER}")
    print(f"GT_SIZE: {GT_SIZE}")
    print(f"生成的gt_sizes: {GT_SIZES}")
    print(f"生成的mini_batch_sizes: {MINI_BATCH_SIZES}")
    print(f"分阶段迭代: {ITERS}")
    print(f"调度器周期: {SCHEDULER['periods']}")
    print("====================")

if __name__ == "__main__":
    print_config()
