"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('log')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    target = 0
    board=""
    data_src = "/media/NAS/nas_187/jaehoon/Fewshot/Patches_SION/TrainValTest_split"
    seed = 1234
    size = 256
    input_size = (size, size)
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 0
    mode = 'test' # 'train' or 'test'
    n_work = 1
    iter_print = True

    if mode == 'train':
        batch_size = 50
        n_iter = 1000
        n_steps = 100

        optim = {
            'lr': 1e-6,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        is_test=True
        snapshot = '/user/home2/soopil/tmp/PANet/runs/PANet_VOC_sets_0_3way_5shot_[train]/2/snapshots/50000.pth'
        n_iter = 1
        n_runs = 1
        n_steps = 1000
        batch_size = 1
    else:
        raise ValueError('Wrong configuration for "mode" !')

    exp_list = [mode]
    exp_str = '_'.join([str(e) for e in exp_list])

    path = {
        'log_dir': './',
        'init_path': './../../pretrained_model/vgg16-397923af.pth',
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook function to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
