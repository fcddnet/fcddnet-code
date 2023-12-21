# -*-coding: utf-8 -*-
"""
    @Project: triple_path_networks
    @File   : gansG.py
    @Author : Lingzhuang Meng
    @E-mail : lzhmeng688@163.com
    @Date   : 2020-01-23 11:18:15
"""
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import pywt
import torchvision
from FCdDNet1 import FCdDNet, ReFCdDNet
from skimage.measure import compare_ssim

import matplotlib.pyplot as plt
from PIL import Image
from numpy import single
from torch.autograd import Variable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils import data
from torchvision import datasets, transforms

torch.cuda.is_available()  # 返回True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_read = False#True  #
root = r"F:\DeepLearning\data\VOC+NET\Testing/"
root512 = r"F:\DeepLearning\data\VOC+NET\Test512"
model_path = r"F:\DeepLearning\data\FCdDNetmodel/"
batch_size = 2
num_epochs = 2000
# 设置学习率
lr1 = 0.001
lr2 = 0.001
u = 0.5


# 数据预览
def imshow(inp, title='Data Show'):  # 若归一化，则要修改方差和均值
    """Imshow for Tensor."""
    # 逆转操作，从 tensor 变回 numpy 数组需要转换通道位置
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


def my_mse_loss(x, y):
    return torch.mean(torch.pow((x - y), 2))


def psnr1(x, y):
    mse = torch.mean(torch.pow((x - y), 2))
    return 10 * torch.log10(255.0 ** 2 / mse)


def PrintF(ind, all, str):
    print("\r{0:^7}\t                 {1:{3}^4}/{2:{3}^4}".format(str, ind, all, chr(12288)), end='')


def Togary(T):
    T1 = T[0, :, :].numpy()
    return T1.astype('float32')


if __name__ == "__main__":
    # 秘密信息
    # coeffs = pywt.dwt2(im, 'haar')
    # -----------------ready the dataset--------------------------
    # 数据读取，将原始数据随机划分为三种数据训练集、验证集合测试集；前两种是用于模型的训练和调整，最后种是用于模型最终的评估
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    data_transforms1 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    # JPEGImages
    all_dataset = datasets.ImageFolder(root, transform=data_transforms)
    all_secret = datasets.ImageFolder(root512, transform=data_transforms1)
    all_dataset_lenth = len(all_dataset)  # 获取所有数据长度
    valid, train = torch.utils.data.random_split(dataset=all_dataset, lengths=[int(all_dataset_lenth * 0.3),
                                                                               all_dataset_lenth - int(
                                                                                   all_dataset_lenth * 0.3)])

    all_secret_lenth = len(all_secret)  # 获取所有数据长度
    valid_secret, train_secret = torch.utils.data.random_split(dataset=all_secret,
                                                               lengths=[int(all_secret_lenth * 0.3),
                                                                        all_secret_lenth - int(
                                                                            all_secret_lenth * 0.3)])

    resNet = FCdDNet()
    print(resNet)
    rresNet = ReFCdDNet()
    print(rresNet)
    # 定义优化方法
    # 采用Cross-Entropy loss,  SGD with moment
    optimizer1 = optim.Adam(resNet.parameters(), lr=lr1)  # , weight_decay=0.75)
    optimizer2 = optim.Adam(rresNet.parameters(), lr=lr2)  # , weight_decay=0.75)
    # 训练模型#

    dataset_sizes = {'train': len(train), 'val': len(valid)}

    # for inputs, labels in dataloaders['train']:
    #     print(inputs.shape)
    resNet = resNet.to(device)
    rresNet = rresNet.to(device)
    if is_read:
        resNet.load_state_dict(
            torch.load(model_path + 'test_FCdDNet-1m' + str(u) + '.pkl'))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
        rresNet.load_state_dict(
            torch.load(model_path + 'test_rFCdDNet-1m' + str(u) + '.pkl'))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
    # 开始训练
    print("Training on ", device)
    time_start = time.time()
    # 训练网络
    # 迭代epoch
    min_loss = torch.Tensor([99999]).to(device)
    for epoch in range(num_epochs):
        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
        trainloader_secret = torch.utils.data.DataLoader(train_secret, batch_size=batch_size, shuffle=True)
        validloader_secret = torch.utils.data.DataLoader(valid_secret, batch_size=batch_size, shuffle=True)
        # 合并便于训练与评估，设置参数
        dataloaders = {
            'train': trainloader,
            'val': validloader
        }
        dataloaders_secret = {
            'train': trainloader_secret,
            'val': validloader_secret
        }
        print('-' * 35)
        print('Epoch {}/{}             {}'.format(epoch + 1, num_epochs,
                                                  time.strftime("%m-%d %H:%M:%S", time.localtime())))
        for phase in ['train', 'val']:
            # 注意训练和验证阶段，需要分别对 model 的设置
            if phase == 'train':
                resNet.train(True)  # 训练模式
                rresNet.train(True)  # 训练模式
            else:
                resNet.train(False)  # 评估模式会关闭Dropout
                rresNet.train(False)  # 评估模式会关闭Dropout
            running_loss = torch.zeros(1).to(device)
            running_loss1 = torch.zeros(1).to(device)
            running_loss2 = torch.zeros(1).to(device)
            start = time.time()
            with torch.no_grad():
                for index, (inputs, _) in enumerate(dataloaders[phase]):
                    imo, _ = iter(dataloaders_secret[phase]).next()
                    im1 = imo[:inputs.shape[0], :, :, :].to(device)

                    Co = inputs.to(device)
                    # 清空参数的梯度
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss1 = torch.zeros(1, requires_grad=True).to(device)
                    loss2 = torch.zeros(1, requires_grad=True).to(device)
                    # 只有训练阶段才追踪历史
                    with torch.set_grad_enabled(phase == 'train'):
                        Ca = resNet(Co, im1)
                        Ca1 = ((Ca.clone() * 255).cpu().detach().numpy()).astype(np.uint8)
                        Ca2 = torch.zeros_like(Ca)
                        # 保存图像再读取，避免原始数据干扰
                        for p in range(inputs.shape[0]):
                            cv2.imwrite("saveImg.jpg", Ca1[p].transpose((1, 2, 0)))
                            img = cv2.imread("saveImg.jpg").transpose((2, 0, 1))
                            Ca2[p] = torch.from_numpy(img.astype(np.float32) / 255).to(device)
                        watermark = rresNet(Ca2)

                        loss1 = my_mse_loss(Co * 255, Ca * 255)  #
                        loss2 = my_mse_loss(im1 * 255, watermark * 255)

                        loss = (loss1) * u + (loss2) * (1 - u)
                        if phase == 'train':
                            loss.backward()  # 最后一次不加retain_graph=True)#
                            optimizer1.step()
                            optimizer2.step()
                        PrintF(index * batch_size + inputs.shape[0], int(dataset_sizes[phase]), phase)
                    # 记录 loss 和 准确率
                    running_loss = running_loss + loss.item()
                    running_loss1 = running_loss1 + loss1.item()
                    running_loss2 = running_loss2 + loss2.item()
                    # running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / (dataset_sizes[phase] / batch_size)
            epoch_loss1 = running_loss1 / (dataset_sizes[phase] / batch_size)
            epoch_loss2 = running_loss2 / (dataset_sizes[phase] / batch_size)
            if phase == 'val' and epoch_loss < min_loss:
                torch.save(resNet.state_dict(), model_path + 'test_FCdDNet-1m' + str(u) + '.pkl',
                           _use_new_zipfile_serialization=False)
                torch.save(rresNet.state_dict(), model_path + 'test_rFCdDNet-1m' + str(u) + '.pkl',
                           _use_new_zipfile_serialization=False)
                min_loss = epoch_loss
            print('\nLoss: {:.4f} \t Loss1: {:.4f} \t Loss2: {:.4f} \t Time: {:.4f} sec'.
                  format(epoch_loss[0], epoch_loss1[0], epoch_loss2[0], time.time() - start))
    print("Training Finshed!!!    Total Time Cost：{:.4f}second!!!".format(time.time() - time_start))
    torch.save(resNet.state_dict(), model_path + 'unet_' + str(u) + '.pkl')
    torch.save(rresNet.state_dict(), model_path + 'reunet_' + str(u) + '.pkl')
