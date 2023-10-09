import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from model.handocc.psp import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class CNNBlock(nn.Module):
    """Base block in CNN"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_act=True, img=True):
        super().__init__()
        self.img = img
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn_act)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn_act)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.img:
            if self.use_bn_act:
                x = self.silu(self.bn2(self.conv2(x)))

                return x
            else:
                return self.silu(self.conv2(x))
        else:
            if self.use_bn_act:
                x = self.silu(self.bn1(self.conv1(x)))

                return x
            else:
                return self.silu(self.conv1(x))
class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = CNNBlock(3, 64, 1, 1, 0, img=False)

        self.e_conv1 = CNNBlock(256, 128, 1, 1, 0, img=False)
        self.e_conv2 = CNNBlock(128, 64, 1, 1, 0, img=False)

        self.conv3 = CNNBlock(128, 256, 1, 1, 0, img=False)

    def forward(self, out_img, x):

        x = self.conv1(x)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)

        emb = self.e_conv1(emb)
        emb = self.e_conv2(emb)

        pointfeat = torch.cat((x, emb), dim=1)

        x = self.conv3(pointfeat)

        bs, dim, _ = x.shape

        x = x.reshape((bs, dim, 32, 32))

        return x