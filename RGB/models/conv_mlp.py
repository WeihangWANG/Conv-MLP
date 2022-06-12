import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from .pre_net import Pre_Net
from .mixer import Mixer

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class ConvMLP(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.pre = Pre_Net(BasicBlock)
        self.mixer = Mixer(BasicBlock)

        self.clr_fc = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(512, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 2))

    def forward(self, x):
        b, n, c, h, w = x.shape

        fea = self.pre.forward(x)
        fea = self.mixer.forward(fea)

        fea = torch.mean(fea, dim=2)
        fea_mean = F.adaptive_avg_pool2d(fea, output_size=1).view(b, -1)
        res = self.clr_fc(fea_mean)

        return res

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
