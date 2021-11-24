import os
from utils import *
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
BatchNorm2d = nn.BatchNorm2d
from model.FaceBagNet_model_B import Pre_Net
from model.mixer import Mixer

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

###########################################################################################3
class FusionNet(nn.Module):
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
        super(FusionNet,self).__init__()

        # self.color_moudle = Net(num_class=num_class)
        # self.ir_moudle = Net(num_class=num_class)
        self.pre = Pre_Net(BasicBlock)
        self.mixer = Mixer(BasicBlock)
        self.depnet = ResNet(BasicBlock, [2,2,2,2], num_classes=2)

        self.clr_fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, num_class))

    ## resnet block for per-parch classification
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def forward(self, x, depth):
    def forward(self, x):
        # x = x[:,:,:2,:,:]
        # print("x: ",x.shape)
        b, n, c, h, w = x.shape
        # b, n, h, w = x.shape
        # x = self.match.forward(x)
        fea = self.pre.forward(x)
        fea = self.mixer.forward(fea)

        fea = torch.mean(fea, dim=2)
        fea_mean = F.adaptive_avg_pool2d(fea, output_size=1).view(b, -1)
        res = self.clr_fc(fea_mean)
        #### softmax
        # res_mix = self.fc(res)
        # #### arcface head
        # kernel_norm = l2_norm(self.kernel, axis=0)
        # res = l2_norm(res, axis=0)
        # res_mix = torch.mm(res, kernel_norm)
        # res_mix = res_mix.clamp(-1, 1)  # for numerical stability
        # # print("res_mix : ",res_mix.size())

        # res_dep = self.depnet(depth)
        #
        # return res, res_dep
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