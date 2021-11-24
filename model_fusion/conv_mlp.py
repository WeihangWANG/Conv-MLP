import os
from utils import *
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
BatchNorm2d = nn.BatchNorm2d
from model.pre_net import Pre_Net
from model.mixer import Mixer

class ConvMLP(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2):
        super(ConvMLP,self).__init__()

        self.pre = Pre_Net(BasicBlock)
        self.mixer = Mixer(BasicBlock)
        self.depnet = ResNet(BasicBlock, [2,2,2,2], num_classes=2)

        self.clr_fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, num_class))

    # def forward(self, x, depth):
    def forward(self, x):

        b, n, c, h, w = x.shape

        fea = self.pre.forward(x)
        fea = self.mixer.forward(fea)

        fea = torch.mean(fea, dim=2)
        fea_mean = F.adaptive_avg_pool2d(fea, output_size=1).view(b, -1)
        res = self.clr_fc(fea_mean)

        return res

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    num_class = 2
    net = Net(num_class)
    print(net)

########################################################################################
if __name__ == '__main__':
    import os
    run_check_net()
    print( 'sucessful!')