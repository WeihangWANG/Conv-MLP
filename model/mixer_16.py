from utils import *
import torchvision.models as tvm
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .backbone.FaceBagNet import FaceBagNet_model_A
BatchNorm2d = nn.BatchNorm2d

###########################################################################################3
class Mixer(nn.Module):
    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict['module.'+key]

        self.load_state_dict(state_dict)
        print('load: '+pretrain_file)

    def __init__(self, block, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Mixer,self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer2 = self._make_layer(block, 128, 3, stride=2)
        self.layer3 = self._make_layer(block, 256, 3, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        # self.post_conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
        #                                      nn.ReLU(inplace=True),
        #                                      nn.BatchNorm2d(128))
        # self.post_conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.BatchNorm2d(256))
        # self.post_conv4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.BatchNorm2d(512))

        # 9 patches 1*1 conv
        self.bottleneck_1 = nn.Sequential(nn.LayerNorm([24, 24]),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        # 9 patches 1*1 conv
        self.bottleneck_2 = nn.Sequential(nn.LayerNorm([12, 12]),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        # 9 patches 1*1 conv
        self.bottleneck_3 = nn.Sequential(nn.LayerNorm([6, 6]),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        # 9 patches 1*1 conv
        self.bottleneck_4 = nn.Sequential(nn.LayerNorm([3, 3]),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.norm = nn.Sequential(nn.LayerNorm([3, 3]),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(512))

        self.shortcut = nn.Sequential(nn.Conv2d(64, 512, kernel_size=1, padding=0, stride=1),
                                      nn.MaxPool2d(8, stride=8),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      )


    def forward(self, x):
        b, n, c, w, h = x.shape
        # print('mimxer input', x.size())
        # cut = x.clone()
        # cut = cut.view(b*n, 64, 24, 24)
        x = x.transpose(1, 2)# print("color_fea size : ", color_fea.size())
        x = x.reshape(b * 64, n, 24, 24)

        x1 = self.bottleneck_1(x)
        # x1 = x1 + x
        # x1 = self.relu(x1)

        x1 = x1.view(b, 64, n, 24, 24)
        x1 = x1.transpose(1, 2)
        x1 = x1.reshape(b * n, 64, 24, 24)

        x1 = self.layer2(x1) #; print('e1',x.size())
        # x1 = self.post_conv2(x1)

        x1 = x1.view(b, n, 128, 12, 12)
        x1 = x1.transpose(1, 2)
        x1 = x1.reshape(b * 128, n, 12, 12)

        x2 = self.bottleneck_2(x1) #; print('e2',x.size())
        # x2 = x2 + x1
        # x2 = self.relu(x2)

        x2 = x2.view(b, 128, n, 12, 12)
        x2 = x2.transpose(1, 2)
        x2 = x2.reshape(b * n, 128, 12, 12)

        x2 = self.layer3(x2)
        # x2 = self.post_conv3(x2)

        x2 = x2.view(b, n, 256, 6, 6)
        x2 = x2.transpose(1, 2)
        x2 = x2.reshape(b * 256, n, 6, 6)

        x3 = self.bottleneck_3(x2)
        # x3 = x3 + x2
        # x3 = self.relu(x3)

        x3 = x3.view(b, 256, n, 6, 6)
        x3 = x3.transpose(1, 2)
        x3 = x3.reshape(b * n, 256, 6, 6)

        x3 = self.layer4(x3)
        # x3 = self.post_conv4(x3)

        # cut = self.shortcut(cut)
        # x4 = x3 + cut
        # x4 = self.norm(x4)
        x4 = x3.view(b, n, 512, 3, 3)
        x4 = x4.transpose(1, 2)
        x4 = x4.reshape(b * 512, n, 3, 3)

        x4 = self.bottleneck_4(x4)

        # x4 = x4 + x3
        # x4 = self.relu(x4)
        x4 = x4.view(b, 512, n, 3, 3)

        return x4

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

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


########################################################################################
if __name__ == '__main__':
    import os
