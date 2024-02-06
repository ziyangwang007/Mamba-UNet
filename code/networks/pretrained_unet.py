# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:08:11 2022

@author: loua2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
import math
import torchvision.models as models
import os

class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class preUnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, **kwargs):
#     def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)



    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)      # bs, 64, 88, 88
        
        # ----------- low-level features -------------
        
        x1 = self.resnet.layer1(x_k)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22

        
        x_up_1 = self.conv_up_1(self.up(x3)) # 512,44,44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1)) #512 ,44,44
        
        x_up_2 = self.conv_up_3(self.up(x_up_1)) # 256,88,88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1)) # 256,88,88

        x_up_3 = self.conv_up_5(self.up(x_up_2)) # 64,176,176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1)) # 64,88,88
        
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)
        return output
