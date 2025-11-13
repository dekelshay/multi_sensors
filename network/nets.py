import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear,MaxPool2d, BatchNorm2d, Dropout
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from torchsummary import summary
from skimage.transform import resize
import copy
from network.spp_layer import spatial_pyramid_pool


class Block(nn.Module):

    def __init__(self,r):
        super(Block, self).__init__()

        self.conv1    = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.BtchNrm1 = nn.BatchNorm2d(32, affine=False)
        #self.SE1 = SE_Block(32, r=4)
            #nn.ReLU(),
        self.GELU1    = nn.ReLU()


        self.conv2    = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.BtchNrm2 = nn.BatchNorm2d(32, affine=False)
        #self.SE2 = SE_Block(32, r=4)
            #nn.ReLU(),
        self.GELU2    = nn.ReLU()


        self.conv3    = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False)  # stride = 2
        self.BtchNrm3 = nn.BatchNorm2d(64, affine=False)
        #self.SE3 = SE_Block(64, r=4)
            #nn.ReLU(),
        self.GELU3 = nn.ReLU()



        self.conv4    = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.BtchNrm4 = nn.BatchNorm2d(64, affine=False)
        self.SE4      = SE_Block(64, r=r)
            #nn.ReLU(),
        self.GELU4    = nn.ReLU()


        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)  # stride = 2  wrong:10368
        self.BtchNrm5 = nn.BatchNorm2d(128, affine=False)
        self.SE5 = SE_Block(128, r=r)
            #nn.ReLU()
        self.GELU5 = nn.ReLU()

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.BtchNrm6 = nn.BatchNorm2d(128, affine=False)
        self.SE6 = SE_Block(128, r=r)
            #nn.ReLU()
        self.GELU6 = nn.ReLU()


        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # BatchSize, 128,8,8
        self.BtchNrm7 = nn.BatchNorm2d(128, affine=False)
        self.SE7      = SE_Block(128, r=r)
            #nn.ReLU(),
        self.GELU7 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.BtchNrm8 = nn.BatchNorm2d(128, affine=False)
        self.SE8      = SE_Block(128, r=r)
        self.GELU8    = nn.ReLU()


    def forward(self, x,UseSE):

        x = self.conv1(x)
        x = self.BtchNrm1(x)
        #x = self.SE1(x)
        # nn.ReLU(),
        x = self.GELU1(x)

        x = self.conv2(x)
        x = self.BtchNrm2(x)
        #x = self.SE2(x)
        # nn.ReLU(),
        x = self.GELU2(x)

        x = self.conv3(x)
        x = self.BtchNrm3(x)
        #x = self.SE3(x)
        # nn.ReLU(),
        x = self.GELU3(x)



        x = self.conv4(x)
        x = self.BtchNrm4(x)
        if UseSE:
            #x = SE_Block_List[3](x)
            x = self.SE4(x)
        # nn.ReLU(),
        x = self.GELU4(x)



        x = self.conv5(x)

        if UseSE:
            #x = SE_Block_List[4](x)
            x = self.SE5(x)
        else:
            x = self.BtchNrm5(x)
        # nn.ReLU(),
        x = self.GELU5(x)


        x = self.conv6(x)
        if UseSE:
            #x = SE_Block_List[5](x)
            x = self.SE6(x)
        else:
            x = self.BtchNrm6(x)
        # nn.ReLU(),
        x = self.GELU6(x)



        x = self.conv7(x)
        if UseSE:
            #x = SE_Block_List[6](x)
            x = self.SE7(x)
        else:
            x = self.BtchNrm7(x)
        # nn.ReLU()
        x = self.GELU7(x)


        x = self.conv8(x)
        if UseSE:
            x = self.SE8(x)
        else:
            x = self.BtchNrm8(x)
        # nn.ReLU()
        #x = self.GELU8(x)

        return x



class Model(nn.Module):

    def __init__(self,DropoutP):
        super(Model, self).__init__()

        self.Block = Block(r=8)

        self.output_num = [8, 4, 2, 1]
        self.output_num = [8]
        self.Dropout = nn.Dropout(DropoutP)

        self.fc1 = nn.Sequential(
            #nn.Linear(10880, 128),
            nn.Linear(8192, 128),
        )

        return

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)


    def FreezeCnn(self, OnOff):
        for param in self.parameters():
            param.requires_grad = not OnOff


    def FreezeBlock(self, OnOff):
        for param in self.block.parameters():
            param.requires_grad = not OnOff


    def forward(self, input1,Mode = 'Normalized',ActivMode =False,UseSE=False):
        BatchSize = input1.size(0)
        feat = self.input_norm(input1)
        feat = self.Block(feat,UseSE)

        spp_a = spatial_pyramid_pool(feat, BatchSize, [int(feat.size(2)), int(feat.size(3))], self.output_num)

        if Mode == 'NoFC':
            return spp_a

        spp_a = self.Dropout(spp_a)  # 20% probability

        feature_a = self.fc1(spp_a).view(BatchSize, -1)

        if Mode ==  'Normalized':
            #return L2Norm()(feature_a)
            if ActivMode:
                return F.normalize(feature_a, dim=1, p=2),feat
            else:
                return F.normalize(feature_a, dim=1, p=2)
        else:
            if ActivMode:
                return feature_a,feat
            else:
                return feature_a








class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.GainFC1 = nn.Linear(c, c // r, bias=True)
        self.GainRelu1 = nn.ReLU(inplace=True)
        self.GainFC2 = nn.Linear(c // r, c // r, bias=True)
        self.GainRelu2 = nn.ReLU(inplace=True)


        self.GainFC3 = nn.Linear(c // r, c, bias=True)
        self.GainSigmd = nn.Sigmoid()



        self.BiasFC1 = nn.Linear(c, c // r, bias=True)
        self.BiasRelu = nn.ReLU(inplace=True)
        self.BiasFC2 = nn.Linear(c // r, c, bias=True)
        self.BiasSigmd = nn.Sigmoid()

    def forward(self, x):
        bs, c, _, _ = x.shape
        y0 = self.squeeze(x).view(bs, c)

        y = self.GainFC1(y0)
        y1 = self.GainRelu1(y)
        #y = self.GainFC2(y)
        #y1 = self.GainRelu2(y)

        y = self.GainFC3(y1)
        y = self.GainSigmd(y).view(bs, c, 1, 1)

        #bias = self.BiasFC1(y0)
        #bias = self.BiasRelu(bias)
        bias = self.BiasFC2(y1)
        bias = bias.view(bs, c, 1, 1)

        return x*y.expand_as(x) + bias.expand_as(x)