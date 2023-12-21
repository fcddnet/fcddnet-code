# -*-coding: utf-8 -*-
# bf2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils import data
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义网络结构


class FCdDNet(torch.nn.Module):
    def __init__(self):
        super(FCdDNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 48, (3, 3), stride=1, padding=1)

        self.bn11 = nn.BatchNorm2d(48)
        self.conv11 = nn.Conv1d(48, 36, (3, 1), stride=1, padding=1)
        self.conv12 = nn.Conv1d(36, 24, (1, 3), stride=1, padding=1)
        self.conv13 = nn.Conv1d(24, 16, (3, 3), stride=1, padding=1)
        self.dp11 = nn.Dropout2d(0.2)

        self.conv1c = nn.Conv2d(64, 128, (1, 1), stride=2, padding=0)
        self.bn1c = nn.BatchNorm2d(128)

        self.conv41 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1)
        self.conv42 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(256)
        self.bn42 = nn.BatchNorm2d(256)

        self.conv2c = nn.Conv2d(128, 256, (1, 1), stride=2, padding=0)
        self.bn2c = nn.BatchNorm2d(256)

        self.conv71 = nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1)
        self.conv72 = nn.ConvTranspose2d(128, 128, (3, 3), stride=1, padding=1)
        self.bn71 = nn.BatchNorm2d(128)
        self.bn72 = nn.BatchNorm2d(128)

        self.conv5c = nn.ConvTranspose2d(256, 128, (2, 2), stride=2, padding=0)
        self.bn5c = nn.BatchNorm2d(128)

        self.conv81 = nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
        self.conv82 = nn.ConvTranspose2d(64, 64, (3, 3), stride=1, padding=1)
        self.bn81 = nn.BatchNorm2d(64)
        self.bn82 = nn.BatchNorm2d(64)

        self.conv6c = nn.ConvTranspose2d(128, 64, (2, 2), stride=2, padding=0)
        self.bn6c = nn.BatchNorm2d(64)

        '''        
        self.conv91 = nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1)
        self.conv92 = nn.ConvTranspose2d(32, 32, (3, 3), stride=1, padding=1)
        self.bn91 = nn.BatchNorm2d(32)
        self.bn92 = nn.BatchNorm2d(32)

        self.conv7c = nn.ConvTranspose2d(64, 32, (2, 2), stride=2, padding=0)
        self.bn7c = nn.BatchNorm2d(32)'''

        self.conv10 = nn.ConvTranspose2d(64, 3, (4, 4), stride=2, padding=1)
        # self.bn10 = nn.BatchNorm2d(3)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        x11 = self.conv1(xy)
        x12 = F.relu(self.bn11(x11))
        x13 = self.dp11(self.conv13(self.conv12(self.conv11(x12))))
        x31 = F.relu(self.bn31(self.conv31(x11)))  #
        x32 = self.bn32(self.conv32(x31))
        shortcut1 = self.bn1c(self.conv1c(x11))
        x33 = F.relu(x32 + shortcut1)  #
        x41 = F.relu(self.bn41(self.conv41(x33)))  #
        x42 = self.bn42(self.conv42(x41))
        shortcut2 = self.bn2c(self.conv2c(x33))
        x43 = F.relu(x42 + shortcut2)  #

        x71 = F.leaky_relu(self.bn71(self.conv71(x43)))  #
        x72 = self.bn72(self.conv72(x71))
        shortcut5 = self.bn5c(self.conv5c(x43))
        x73 = F.leaky_relu(x72 + shortcut5)  #
        x81 = F.leaky_relu(self.bn81(self.conv81(x73)))  #
        x82 = self.bn82(self.conv82(x81))
        shortcut6 = self.bn6c(self.conv6c(x73))
        x83 = F.leaky_relu(x82 + shortcut6)  #
        '''x91 = F.leaky_relu(self.bn91(self.conv91(x83)))  #
        x92 = self.bn92(self.conv92(x91))
        #shortcut7 = self.bn7c(self.conv7c(x83))
        x93 = F.leaky_relu(x92+shortcut7)  #
        '''
        out = torch.sigmoid(self.conv10(x83))
        return out

    '''net = UNet()
    img = images[1,:,:,:]
    img = torch.cat([img[np.newaxis, :, :, :], Bgray[np.newaxis, :, :, :]], dim=1)
    y = net(img)
    print(y.shape)
    '''


class RFCdDNet(nn.Module):
    def __init__(self):
        super(RFCdDNet, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, (3, 3), stride=2, padding=1)
        # self.bn11 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)
        self.conv22 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv1c = nn.Conv2d(64, 128, (1, 1), stride=2, padding=0)
        self.bn1c = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1)
        self.conv32 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm2d(256)

        self.conv2c = nn.Conv2d(128, 256, (1, 1), stride=2, padding=0)
        self.bn2c = nn.BatchNorm2d(256)

        self.convt41 = nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1)
        self.convt42 = nn.ConvTranspose2d(128, 128, (3, 3), stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.bn42 = nn.BatchNorm2d(128)

        self.conv5c = nn.ConvTranspose2d(256, 128, (2, 2), stride=2, padding=0)
        self.bn5c = nn.BatchNorm2d(128)

        self.convt51 = nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
        self.convt52 = nn.ConvTranspose2d(64, 64, (3, 3), stride=1, padding=1)
        self.bn51 = nn.BatchNorm2d(64)
        self.bn52 = nn.BatchNorm2d(64)

        self.conv6c = nn.ConvTranspose2d(128, 64, (2, 2), stride=2, padding=0)
        self.bn6c = nn.BatchNorm2d(64)

        self.convt61 = nn.ConvTranspose2d(64, 3, (4, 4), stride=2, padding=1)
        # self.bn61 = nn.BatchNorm2d(3)

    def forward(self, x):
        x11 = F.relu(self.conv11(x))
        x21 = F.relu(self.bn21(self.conv21(x11)))
        x22 = F.relu(self.bn22(self.conv22(x21)) + self.bn1c(self.conv1c(x11)))
        x31 = F.relu(self.bn31(self.conv31(x22)))
        x32 = F.relu(self.bn32(self.conv32(x31)) + self.bn2c(self.conv2c(x22)))
        x41 = F.leaky_relu(self.bn41(self.convt41(x32)))
        x42 = F.leaky_relu(self.bn42(self.convt42(x41)) + self.bn5c(self.conv5c(x32)))
        x51 = F.leaky_relu(self.bn51(self.convt51(x42)))
        x52 = F.leaky_relu(self.bn52(self.convt52(x51)) + self.bn6c(self.conv6c(x42)))
        out = torch.sigmoid(self.convt61(x52))  # torch.sigmoid(self.conv6(x5))
        return out
