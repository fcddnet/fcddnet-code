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


class _DenseDilatedLayer(nn.Sequential):
    '''DenseBlock中的内部结构，这里是BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv结构'''
    def __init__(self, in_dim, out_dim, rates, kernel=3, bias=False, extend_dim=False):
        super(_DenseDilatedLayer, self).__init__()
        self.features = []
        self.num = len(rates)
        self.in_dim = in_dim
        self.out_dim = out_dim
        for idx, rate in enumerate(rates):
            self.features.append(nn.Sequential(
                nn.Conv2d(self.in_dim + idx * out_dim,
                          out_dim,
                          kernel_size=kernel, dilation=rate,
                          padding=rate * (kernel - 1) // 2, bias=bias),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
            )

        self.features = nn.ModuleList(self.features)

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(self.in_dim + out_dim * self.num,
                      self.out_dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        for f in self.features:
            x = torch.cat([f(x), x], 1)
        x = self.conv1x1_out(x)
        return x


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm1', nn.BatchNorm2d(num_output_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.add_module('relu2', nn.ReLU())

class _Layer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Layer, self).__init__()
        self.add_module('conv1', nn.Conv2d(num_input_features, int((num_input_features+num_output_features)/2),
                                          kernel_size=(3,1), stride=1, padding=1, dilation=2, bias=False))
        self.add_module('norm1', nn.BatchNorm2d(int((num_input_features+num_output_features)/2)))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(int((num_input_features+num_output_features)/2), num_output_features,
                                          kernel_size=(1,3), stride=1, padding=1, dilation=2, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.add_module('relu2', nn.ReLU())

class _DeConv(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_DeConv, self).__init__()
        self.add_module('deconv', nn.ConvTranspose2d(num_input_features, num_output_features,
                                          kernel_size=(3,3), stride=2, padding=1, output_padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('leaky_relu', nn.LeakyReLU())

class FCdDNet(nn.Module):
    def __init__(self):
        super(FCdDNet, self).__init__()
        self.trans0 = _Transition(num_input_features=3, num_output_features=6)

        self.conv1 = nn.Conv2d(9, 48, (3, 3), stride=1, padding=1)
        self.bnc1 = nn.BatchNorm2d(48)
        self.lay1 = _Layer(48,16)

        self.trans1 = _Transition(num_input_features=48+16, num_output_features=64)
        self.lay2 = _Layer(64,32)

        self.trans2 = _Transition(num_input_features=32+64, num_output_features=96)
        self.ddblock = _DenseDilatedLayer(in_dim=96, out_dim=64, rates=[2,4,8,16], kernel=3, bias=False, extend_dim=False)#num_layers=3, num_input_features=256, bn_size=2, growth_rate=16)

        self.trans3 = _Transition(num_input_features=160, num_output_features=160)
        self.lay3 = _Layer(160,16)
        self.deconv1 = _DeConv(16, 176)

        self.lay4 = _Layer(176+160,16)
        self.deconv2 = _DeConv(16, 112)

        self.lay5 = _Layer(112+96,16)
        self.deconv3 = _DeConv(16, 80)

        self.lay6 = _Layer(80+64,16)
        self.conv2 = nn.Conv2d(16, 3, (1, 1), stride=1, padding=0)

    def forward(self, x, y):
        # cA, cH, cV, cD = coeffs
        y0 = self.trans0(y)
        xy = torch.cat([x, y0], dim=1)
        x11 = F.relu(self.bnc1(self.conv1(xy)))  # 6
        x12 = self.lay1(x11)
        merge1 = torch.cat([x11, x12], dim=1)
        x21 = self.trans1(merge1)
        x22 = self.lay2(x21)
        merge2 = torch.cat([x21, x22], dim=1)
        x31 = self.trans2(merge2)
        x32 = self.ddblock(x31)

        merge3 = torch.cat([x31, x32], dim=1)
        x41 = self.trans3(merge3)
        x42 = self.lay3(x41)
        x43 = self.deconv1(x42)

        merge4 = torch.cat([merge3, x43], dim=1)
        x51 = self.lay4(merge4)
        x52 = self.deconv2(x51)
        merge5 = torch.cat([merge2, x52], dim=1)
        x61 = self.lay5(merge5)
        x62 = self.deconv3(x61)
        merge6 = torch.cat([merge1, x62], dim=1)
        x71 = self.lay6(merge6)
        x72 = F.relu(self.conv2(x71))
        return x72


'''net = UNet()
img = images[1,:,:,:]
img = torch.cat([img[np.newaxis, :, :, :], Bgray[np.newaxis, :, :, :]], dim=1)
y = net(img)
print(y.shape)
'''


class ReFCdDNet(nn.Module):
    def __init__(self):
        super(ReFCdDNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, (3, 3), stride=1, padding=1)
        self.bnc1 = nn.BatchNorm2d(48)
        self.lay1 = _Layer(48, 16)

        self.trans1 = _Transition(num_input_features=48 + 16, num_output_features=64)
        self.lay2 = _Layer(64, 32)

        self.trans2 = _Transition(num_input_features=32 + 64, num_output_features=96)
        self.ddblock = _DenseDilatedLayer(in_dim=96, out_dim=64, rates=[2, 4, 8, 16], kernel=3, bias=False,
                                          extend_dim=False)  # num_layers=3, num_input_features=256, bn_size=2, growth_rate=16)

        self.trans3 = _Transition(num_input_features=160, num_output_features=160)
        self.lay3 = _Layer(160, 16)
        self.deconv1 = _DeConv(16, 176)

        self.lay4 = _Layer(176 + 160, 16)
        self.deconv2 = _DeConv(16, 112)

        self.lay5 = _Layer(112 + 96, 16)
        self.deconv3 = _DeConv(16, 80)

        self.lay6 = _Layer(80 + 64, 16)
        self.deconv4 = _DeConv(16, 3)
        self.conv2 = nn.ConvTranspose2d(16, 3, (3, 3), stride=2, padding=1)

    def forward(self, x):
        # cA, cH, cV, cD = coeffs
        x11 = F.relu(self.bnc1(self.conv1(x)))  # 6
        x12 = self.lay1(x11)
        merge1 = torch.cat([x11, x12], dim=1)
        x21 = self.trans1(merge1)
        x22 = self.lay2(x21)
        merge2 = torch.cat([x21, x22], dim=1)
        x31 = self.trans2(merge2)
        x32 = self.ddblock(x31)

        merge3 = torch.cat([x31, x32], dim=1)
        x41 = self.trans3(merge3)
        x42 = self.lay3(x41)
        x43 = self.deconv1(x42)

        merge4 = torch.cat([merge3, x43], dim=1)
        x51 = self.lay4(merge4)
        x52 = self.deconv2(x51)
        merge5 = torch.cat([merge2, x52], dim=1)
        x61 = self.lay5(merge5)
        x62 = self.deconv3(x61)
        merge6 = torch.cat([merge1, x62], dim=1)
        x71 = self.lay6(merge6)
        x72 = F.leaky_relu(self.deconv4(x71))
        return x72