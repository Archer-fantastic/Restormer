import argparse
import datetime
import logging
import math
import random
import time
import torch
import sys
import os
from os import path as osp

# Add project root to sys.path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            
            if num_iter_per_epoch == 0:
                print(f"Warning: num_iter_per_epoch is 0. len(train_set)={len(train_set)}, batch_size_per_gpu={dataset_opt['batch_size_per_gpu']}")
                num_iter_per_epoch = 1

            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        # 优先使用 last.state
        if 'last.state' in states:
            resume_state = os.path.join(state_folder_path, 'last.state')
            opt['path']['resume_state'] = resume_state
        # 其次使用 latest.state
        elif 'latest.state' in states:
            resume_state = os.path.join(state_folder_path, 'latest.state')
            opt['path']['resume_state'] = resume_state
        # 再次使用 best.state
        elif 'best.state' in states:
            resume_state = os.path.join(state_folder_path, 'best.state')
            opt['path']['resume_state'] = resume_state
        else:
            # 过滤出数字开头的状态文件
            numeric_states = []
            for state_file in states:
                if state_file.endswith('.state'):
                    # 移除 .state 后缀
                    state_name = state_file[:-6]
                    # 检查是否是数字
                    if state_name.isdigit():
                        numeric_states.append(state_file)
            
            if len(numeric_states) > 0:
                # 找到最大的数字状态文件
                max_state_file = max(numeric_states, key=lambda x: int(x[:-6]))
                resume_state = os.path.join(state_folder_path, max_state_file)
                opt['path']['resume_state'] = resume_state
            else:
                # 没有数字状态文件，使用最后修改时间最近的文件
                states.sort(key=lambda x: os.path.getmtime(os.path.join(state_folder_path, x)), reverse=True)
                if states:
                    resume_state = os.path.join(state_folder_path, states[0])
                    opt['path']['resume_state'] = resume_state

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
        # 检查 current_iter 是否为字符串
        if isinstance(resume_state.get('iter'), str):
            # 如果是 'best' 或其他字符串，设置为 0 或其他合理值
            resume_state['iter'] = 0
            logger.warning('Resume state has string iter value, setting to 0')
    else:
        resume_state = None

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # 初始化最佳模型跟踪
    best_metric = -float('inf')
    best_iter = -1

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
            ### ------Progressive learning ---------------------
            try:
                # 计算当前阶段索引
                j = ((current_iter>groups) !=True).nonzero()
                if len(j) == 0:
                    bs_j = len(groups) - 1
                else:
                    bs_j = j[0][0]

                mini_gt_size = mini_gt_sizes[bs_j]
                mini_batch_size = mini_batch_sizes[bs_j]
                
                if logger_j[bs_j]:
                    logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count()))
                    logger.info(f'Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
                    logger_j[bs_j] = False

                lq = train_data['lq']
                gt = train_data['gt']

                if mini_batch_size < batch_size:
                    indices = random.sample(range(0, batch_size), k=mini_batch_size)
                    lq = lq[indices]
                    gt = gt[indices]
                    # 清理原始大张量
                    torch.cuda.empty_cache()

                if mini_gt_size < gt_size:
                    # 确保裁剪区域在有效范围内
                    max_offset = gt_size - mini_gt_size
                    if max_offset > 0:
                        x0 = int(max_offset * random.random())
                        y0 = int(max_offset * random.random())
                        x1 = x0 + mini_gt_size
                        y1 = y0 + mini_gt_size
                        lq = lq[:,:,x0:x1,y0:y1]
                        gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
                        # 清理原始大张量
                        torch.cuda.empty_cache()
                # 最后清理一次内存
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f'Error during progressive learning: {e}')
                # 清理内存
                torch.cuda.empty_cache()
                # 继续训练，使用默认值
                mini_gt_size = gt_size
                mini_batch_size = batch_size
            ###-------------------------------------------

            
            try:
                model.feed_train_data({'lq': lq, 'gt':gt})
                model.optimize_parameters(current_iter)
            except Exception as e:
                logger.error(f'Error during optimize_parameters at iter {current_iter}: {e}')
                import traceback
                logger.error(traceback.format_exc())
                raise e
            
            # 清理模型输入数据
            del lq, gt
            torch.cuda.empty_cache()

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if opt['logger']['save_checkpoint_freq'] > 0 and current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                # 保存当前iter的模型和状态
                model.save(epoch, current_iter, True)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                # 验证并获取指标结果
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                
                # 检查是否是最佳模型
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_iter = current_iter
                    logger.info(f'New best model found at iter {current_iter}, metric: {best_metric:.4f}')
                    # 保存最佳模型
                    model.save(epoch, 'best')
                    # 保存当前iter的模型和状态
                    model.save(epoch, current_iter, True)
                else:
                    # 每个验证周期结束后保存当前iter的模型和状态
                    logger.info('Saving current model and state.')
                    model.save(epoch, current_iter, True)
                
                # 验证后清理内存
                torch.cuda.empty_cache()

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
