import os
import re
import sys
import random
import numpy as np
from glob import glob
from cv2 import resize
from PIL import Image
import torch

def path_collect(cfg, task_idx, option='train'):
    normal_paths = glob(f"{cfg['data_src']}/{cfg['organs'][task_idx]}/{option}/normal/*")
    tumor_paths = glob(f"{cfg['data_src']}/{cfg['organs'][task_idx]}/{option}/tumor/*")
    return normal_paths, tumor_paths

def train_loader(cfg, transform):
    print(">> setting dataloaders.")
    trg_task = cfg["target"]

    if cfg["is_exception"]:
        tasks = [0,1,2,3,4]
        if not cfg["is_use_all"]:
            tasks.remove(trg_task)
        normals, tumors = [],[]
        for task in tasks:
            n, t = path_collect(cfg, task, option='train')
            normals += n
            tumors += t
    else:
        normals, tumors = path_collect(cfg, trg_task, option='train')
    return TrainLoader(normals, tumors, cfg, transform, is_train=True)

def test_loader(cfg, transforms, option='test', is_finetuning=False):
    trg_task = cfg["target"]
    normals, tumors = path_collect(cfg, trg_task, option=option)
    test_dataset = TestLoader(normals, tumors, cfg, transforms, is_train=False)
    return test_dataset

class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, normal_paths, tumor_paths, cfg, transforms, is_train):
        # super().__init__()
        self.is_train = is_train
        self.mode = cfg['mode']
        self.n_data_use = cfg['n_iter']
        self.normal_paths = normal_paths
        self.n_subj = len(self.normal_paths)
        self.subj_idx_space = [i for i in range(self.n_subj)]
        self.tumor_paths = tumor_paths
        self.transforms = transforms

        ## load file names in advance
        self.n_normal_img, self.n_tumor_img = 0, 0
        self.normal_list, self.tumor_list = [], []
        self.normal_nums, self.tumor_nums = [], []
        for img_path in self.normal_paths:
            l = os.listdir(img_path)
            self.n_normal_img += len(l)
            self.normal_nums.append(len(l))
            self.normal_list.append(l)

        for img_path in self.tumor_paths:
            l = os.listdir(img_path)
            self.n_tumor_img += len(l)
            self.tumor_nums.append(len(l))
            self.tumor_list.append(l)

    def get_label(self, path):
        dir_name = path.split("/")[-3]
        if dir_name == "normal":
            return 0
        elif dir_name == "tumor":
            return 1
        else:
            print("wrong label configuration.")
            assert False

    def get_sample(self, path):
        # seed = random.randrange(0,1000)
        im = Image.open(path)
        input = self.transforms(im)
        label = self.get_label(path)
        sample = {
            "x":input.type(torch.float32)/255,
            "y":label,
            "path":path,
        }
        return sample

class TrainLoader(BaseLoader):
    def __init__(self, normal_paths, tumor_paths, cfg, transforms, is_train):
        super().__init__(normal_paths, tumor_paths, cfg, transforms, is_train)

    def __len__(self):
        if self.is_train:
            return self.n_data_use

    def __getitem__(self, item):
        ## sample subjects first
        subj_idx = random.sample(self.subj_idx_space, 1)[0]
        # print(subj_idx)
        ## normal (0) / tumor (1) choose
        choose = random.sample([0,1],1)[0]
        if choose == 0:
            file = random.sample(self.normal_list[subj_idx],1)[0]
            path = f"{self.normal_paths[subj_idx]}/{file}"
        else:
            file = random.sample(self.tumor_list[subj_idx],1)[0]
            path = f"{self.tumor_paths[subj_idx]}/{file}"

        return self.get_sample(path)

class TestLoader(BaseLoader):
    def __init__(self, normal_paths, tumor_paths, cfg, transforms, is_train):
        super().__init__(normal_paths, tumor_paths, cfg, transforms, is_train)

        self.all_test_files = []
        for subj_path in self.normal_paths:
            files = os.listdir(subj_path)
            for file in files:
                self.all_test_files.append(f"{subj_path}/{file}")

        for subj_path in self.tumor_paths:
            files = os.listdir(subj_path)
            for file in files:
                self.all_test_files.append(f"{subj_path}/{file}")

        # for e in self.all_test_files:
        #     print(e)

    def __len__(self):
        return len(self.all_test_files)

    def __getitem__(self, idx):
        path = self.all_test_files[idx]
        return self.get_sample(path)

if __name__ == "__main__":
    pass
    # main()