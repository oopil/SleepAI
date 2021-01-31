"""Evaluation Script"""
import os
import shutil
import pdb
# import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torchvision.transforms import Compose, transforms, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, GaussianBlur, Normalize
from torch.autograd import Variable
from util.utils import *
from config import ex
from tensorboardX import SummaryWriter
import time

if __name__=='__main__':
    from dataloaders.data import *
    from model import *
else:
    from .dataloaders.data import *
    from .model import *

def count_parameters(model):
    ## count # of parameters
    parameters = list(model.parameters())
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print(f"# of parameters : {pp}")

@ex.automain
def main(_run, _config, _log):
    torch.multiprocessing.set_sharing_strategy('file_system')
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{_config['gpu_id']}")
    _log.info('###### Load data ######')

    transform_test = Compose([
        Resize(size=_config['input_size']),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = test_loader(_config, transform_test, option='test')

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        # batch_size=_config['batch_size'],
        shuffle=False,
        num_workers=_config['n_work'],
        pin_memory=False,#True
        drop_last=False
    )

    _log.info('###### Create model ######')
    model = Classifier(_config).to(device)
    checkpoint = torch.load(_config['snapshot'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if _config['record']:
        _log.info('###### define tensorboard writer #####')
        board_name = f'board/test_{_config["board"]}_{date()}'
        writer = SummaryWriter(board_name)

    _log.info('###### Testing begins ######')

    def predict(model, sample, batch=1):
        x = sample['x'].to(device)  # [B, way, shot, 3, 256, 256]
        pred = model(x)
        return pred

    results = []
    model.eval()
    with torch.no_grad():
        for i, sample_test in enumerate(testloader): # even for upward, down for downward
            # if i == 10: break
            pred = predict(model, sample_test, batch=1)
            sample_test['pred'] = pred.cpu()
            results.append(sample_test)
            # y, pred.argmax(dim=1).cpu().numpy()
            if _config["iter_print"]:
                print(f"test, iter:{i}/{len(testloader)}\t\t", end='\r')

    print("computing performances ... total ", len(results))
    correct = 0
    for result in results:
        y = result['y']
        pred = result['pred']
        path = result['path'][0]
        path.split('/')
        if y == pred.argmax(dim=1):
            correct+=1

    print(f"accuracy : {100*correct/len(results)}")
