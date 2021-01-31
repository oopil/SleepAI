"""Training Script"""
import os
import shutil
import numpy as np
import pdb
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose, transforms,\
    ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, \
    GaussianBlur, Normalize, Scale
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.autograd import Variable

if __name__ == '__main__':
    from util.utils import *
    from dataloaders.data import *
    from model import *
    from config import ex
else:
    from .util.utils import *
    from .config import ex
    from .dataloaders.data import *
    from .model import *

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(f"cuda:{_config['gpu_id']}")

    _log.info('###### Load data ######')
    transform_train = Compose([ #GaussianSmoothing([0, 5]),
        Resize(size=_config['input_size']),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ToTensor(),
        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                          ])

    transform_test = Compose([
        Resize(size=_config['input_size']),
        ToTensor(),
        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ])

    tr_dataset = train_loader(_config, transform_train)
    val_dataset = test_loader(_config, transform_test, option='valid')

    trainloader = DataLoader(
        dataset=tr_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False, #True load data while training gpu
        drop_last=True
    )
    validationloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        # batch_size=_config['batch_size'],
        shuffle=False,
        num_workers=_config['n_work'],
        pin_memory=False,#True
        drop_last=False
    )

    model = Classifier(_config).to(device)
    _log.info('###### Set optimizer ######')
    print(_config['optim'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 _config['optim']['lr'])
    # scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCELoss()
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_func = nn.MSELoss()

    min_val_loss = 100000.0
    min_epoch = 0
    iter_n_train, iter_n_val = len(trainloader), len(validationloader)
    blank = torch.zeros([1, 256, 256]).to(device)

    def predict(model_, sample, loss_func_, batch=1):
        x = sample['x'].to(device)  # [B, way, shot, 3, 256, 256]
        y = sample['y'].to(device).type(torch.int64)  # [B]
        pred = model_(x)
        loss = loss_func_(pred, y) # squeeze for BCE loss
        # print(pred.cpu().detach().numpy())
        return pred, loss

    _log.info('###### Training ######')
    # inputs = set()
    for i_epoch in range(_config['n_steps']):
        model.train()
        train_correct, valid_correct = 0, 0
        loss_epoch = 0
	## training stage
        for i_iter, sample_train in enumerate(trainloader):    
            optimizer.zero_grad()
            pred, loss = predict(model, sample_train, loss_func, batch=_config["batch_size"])
            loss.backward()
            optimizer.step()
            loss_epoch += loss/iter_n_train
            train_correct += torch.sum(torch.eq(sample_train['y'],pred.argmax(dim=1).cpu()))
            if _config["iter_print"]:
                print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss}", end='\r')

	## validation stage
        model.eval()
        with torch.no_grad(): 
            loss_valid = 0
            for i_iter, sample_valid in enumerate(validationloader):
                pred, loss = predict(model, sample_valid, loss_func, batch=_config["batch_size"])
                loss_valid += loss/iter_n_val
                if sample_valid['y'] == pred.argmax(dim=1).cpu():
                    valid_correct += 1
                if _config["iter_print"]:
                    print(f"valid, iter:{i_iter}/{iter_n_val}, iter_loss:{loss}", end='\r')

        train_accur = tensor2float(train_correct/_config['n_iter'],3)
        valid_accur = tensor2float(valid_correct/iter_n_val,3)
        loss_epoch, loss_valid = tensor2float(loss_epoch, 5), tensor2float(loss_valid, 5)

        if min_val_loss > loss_valid:
            min_epoch = i_epoch
            min_val_loss = loss_valid
            print(f"train - epoch:{i_epoch}/{_config['n_steps']}| epoch_loss:{loss_epoch}| valid_loss:{loss_valid}| train_accur : {train_accur}| valid_accur : {valid_accur} => model saved", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/lowest.pth'
        else:
            print(f"train - epoch:{i_epoch}/{_config['n_steps']}| epoch_loss:{loss_epoch}| valid_loss:{loss_valid}| train_accur : {train_accur}| valid_accur : {valid_accur} - min epoch:{min_epoch}", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'

        _run.log_scalar("training.loss", float(loss_epoch), i_epoch)
        _run.log_scalar("validation.loss", float(loss_valid), i_epoch)
        _run.log_scalar("min_epoch", min_epoch, i_epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_fname
        )
    print(f"{_run.experiment_info['name']} - ID : {_run._id}")
