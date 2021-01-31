import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import pdb

class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.ch = 512
        self.in_size = cfg["size"]
        self.reduced_size = self.in_size/(2**5)
        self.flatten = nn.Flatten()
        flat_ch = int(self.ch*self.reduced_size*self.reduced_size)
        # print(self.in_size, flat_ch)

        self.fcl = nn.Sequential(
            nn.Linear(flat_ch, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True),
            nn.Softmax(dim=1),
        )
        self._init_weights()
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # torch.nn.init.normal(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.encoder(x)
        # print(out.shape)
        out = self.flatten(out)
        out = self.fcl(out)
        return out
