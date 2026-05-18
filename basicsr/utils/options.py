import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    import os
    import importlib.util
    
    # 检查文件扩展名
    ext = os.path.splitext(opt_path)[1].lower()
    print(f"解析配置文件: {opt_path}")
    print(f"文件扩展名: {ext}")
    
    if ext in ['.yaml', '.yml']:
        # YAML文件解析
        print("使用YAML解析器")
        with open(opt_path, mode='r', encoding='utf-8') as f:
            Loader, _ = ordered_yaml()
            opt = yaml.load(f, Loader=Loader)
    elif ext == '.py':
        # Python文件解析
        print("使用Python解析器")
        try:
            # 使用绝对路径
            abs_opt_path = os.path.abspath(opt_path)
            print(f"加载Python模块: {abs_opt_path}")
            print(f"文件存在: {os.path.exists(abs_opt_path)}")
            
            spec = importlib.util.spec_from_file_location("config", abs_opt_path)
            if spec is None:
                raise ValueError(f"无法加载配置文件: {abs_opt_path}")
            
            config_module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ValueError(f"无法加载配置文件: {abs_opt_path}")
            
            print("执行Python模块")
            spec.loader.exec_module(config_module)
            
            # 构建配置字典
            print("构建配置字典")
            opt = {
                'name': getattr(config_module, 'NAME', 'default'),
                'model_type': getattr(config_module, 'MODEL_TYPE', 'ImageCleanModel'),
                'scale': getattr(config_module, 'SCALE', 1),
                'num_gpu': getattr(config_module, 'NUM_GPU', 1),
                'manual_seed': getattr(config_module, 'MANUAL_SEED', 100),
                'datasets': getattr(config_module, 'DATASETS', {}),
                'network_g': getattr(config_module, 'NETWORK_G', {}),
                'path': getattr(config_module, 'PATH', {}),
                'train': {
                    'total_iter': getattr(config_module, 'TOTAL_ITER', 100000),
                    'warmup_iter': getattr(config_module, 'WARMUP_ITER', 500),
                    'use_grad_clip': getattr(config_module, 'USE_GRAD_CLIP', True),
                    'use_amp': getattr(config_module, 'USE_AMP', False),
                    'amp_level': getattr(config_module, 'AMP_LEVEL', 'O1'),
                    'scheduler': getattr(config_module, 'SCHEDULER', {}),
                    'mixing_augs': getattr(config_module, 'MIXING_AUGS', {}),
                    'optim_g': getattr(config_module, 'OPTIM_G', {}),
                    'pixel_opt': getattr(config_module, 'PIXEL_OPT', {})
                },
                'val': getattr(config_module, 'VAL', {}),
                'logger': getattr(config_module, 'LOGGER', {}),
                'dist_params': getattr(config_module, 'DIST_PARAMS', {})
            }
            print(f"配置名称: {opt['name']}")
            print(f"总迭代数: {opt['train']['total_iter']}")
        except Exception as e:
            print(f"解析Python配置文件失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only .yaml, .yml and .py are supported.")

    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
