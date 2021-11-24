from utils import *
import torchvision.models as tvm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d


###########################################################################################3
class Pre_Net(nn.Module):
    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict['module.'+key]

        self.load_state_dict(state_dict)
        print('load: '+pretrain_file)

    def __init__(self, block, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, is_first_bn=True):
        super(Pre_Net,self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.is_first_bn = is_first_bn
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=5, stride=2, padding=2,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, 2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
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

    def forward(self, x):
        b, n, c, h, w = x.shape
        # b, n, h, w = x.shape
        x = x.view(b * n, c, w, h)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x1 = x.view(b, n, 64, 24, 24)

        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x1

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
