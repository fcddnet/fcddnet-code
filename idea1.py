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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import pywt
import torchvision
from FCdDNet import FCdDNet, RFCdDNet
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
root = r"F:\DeepLearning\data\VOC2007/"
model_path = r"F:\DeepLearning\data\ResNetmodel/"
root_watermark = r"F:\DeepLearning\data/gray/"
#watermark_path = r'F:\DeepLearning\/data/2007_000241.jpg'#r'F:\DeepLearning\/data/2007_009550.png'  # 01.png'
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


def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)
    if imageA.shape == (3, 256, 256):
        # 通道分离，注意顺序BGR不是RGB
        (B1, G1, R1) = imageA[:, :, :]
        (B2, G2, R2) = imageB[:, :, :]
        (score0, diffB) = compare_ssim(B1, B2, full=True)
        (score1, diffG) = compare_ssim(G1, G2, full=True)
        (score2, diffR) = compare_ssim(R1, R2, full=True)
        aveScore = (score0 + score1 + score2) / 3
    else:
        (aveScore, diff) = compare_ssim(imageA, imageB, full=True)
    return aveScore


def Togary(T):
    T1 = T[0, :, :].numpy()
    return T1.astype('float32')


if __name__ == "__main__":
    # 秘密信息
    #im = plt.imread(watermark_path)
    #imo = torchvision.transforms.ToTensor()(im)
    #coeffs = pywt.dwt2(im, 'haar')
    # -----------------ready the dataset--------------------------
    # 数据读取，将原始数据随机划分为三种数据训练集、验证集合测试集；前两种是用于模型的训练和调整，最后种是用于模型最终的评估
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    # JPEGImages
    all_dataset = datasets.ImageFolder(root + 'test', transform=data_transforms)
    all_dataset_lenth = len(all_dataset)  # 获取所有数据长度

    train, test, valid = torch.utils.data.random_split(dataset=all_dataset,
                                                       lengths=[int(all_dataset_lenth * 0.8),
                                                                int(all_dataset_lenth * 0.15),
                                                                all_dataset_lenth - int(all_dataset_lenth * 0.8) - int(
                                                                    all_dataset_lenth * 0.15)])
    # len(valid)+len(train)+len(test) == len(all_dataset)#验证一下数据长度是否一致
    # 数据加载
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

    fcddnet = FCdDNet()
    print(fcddnet)
    rfcddnet= FCdDNet()
    print(rfcddnet)
    # 定义优化方法
    # 采用Cross-Entropy loss,  SGD with moment
    optimizer1 = optim.Adam(fcddnet.parameters(), lr=lr1)  # , weight_decay=0.75)
    optimizer2 = optim.Adam(rfcddnet.parameters(), lr=lr2)  # , weight_decay=0.75)
    # 训练模型#

    # 合并便于训练与评估，设置参数
    dataloaders = {
        'train': trainloader,
        'val': validloader
    }
    dataset_sizes = {'train': len(train), 'val': len(valid)}

    # for inputs, labels in dataloaders['train']:
    #     print(inputs.shape)
    fcddnet = fcddnet.to(device)
    rfcddnet = rfcddnet.to(device)
    if is_read:
        fcddnet.load_state_dict(torch.load(model_path + 'best_resNet-4c.pkl'))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
        rfcddnet.load_state_dict(torch.load(model_path + 'best_rresNet-4c.pkl'))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
       # 开始训练
    print("Training on ", device)
    time_start = time.time()
    # 训练网络
    # 迭代epoch
    min_loss = torch.Tensor([99999]).to(device)
    for epoch in range(num_epochs):
        print('-' * 35)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            # 注意训练和验证阶段，需要分别对 model 的设置
            if phase == 'train':
                fcddnet.train(True)  # 训练模式
                rfcddnet.train(True)  # 训练模式
            else:
                fcddnet.train(False)  # 评估模式会关闭Dropout
                rfcddnet.train(False)  # 评估模式会关闭Dropout
            running_loss = torch.zeros(math.ceil(dataset_sizes[phase]/batch_size)).to(device)
            running_loss1 = torch.zeros(math.ceil(dataset_sizes[phase]/batch_size)).to(device)
            running_loss2 = torch.zeros(math.ceil(dataset_sizes[phase]/batch_size)).to(device)
            start = time.time()
            with torch.no_grad():
                for index, (inputs, _) in enumerate(dataloaders[phase]):
                    #im1 = torch.repeat_interleave(imo.unsqueeze(0), repeats=inputs.shape[0], dim=0).to(device)

                    im0, _ = iter(trainloader).next()
                    im1 = im0[:inputs.shape[0],:,:,:].to(device)
                    '''im = Togary(imo[0])
                    coeffs = pywt.dwt2(im, 'haar')
                    '''
                    Co = inputs.to(device)
                    # 清空参数的梯度
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss1 = torch.zeros(1, requires_grad=True).to(device)
                    loss2 = torch.zeros(1, requires_grad=True).to(device)
                    # 只有训练阶段才追踪历史
                    with torch.set_grad_enabled(phase == 'train'):
                        Ca = fcddnet(Co, im1)
                        watermark = rfcddnet(Ca)
                        #a, (h, v, d) = rresNet(Ca)
                        #watermark = np.ones((inputs.shape[0], 256, 256), dtype=np.float)

                        #for i in range(inputs.shape[0]):
                        '''
                            watermark[i] = pywt.idwt2((a[i][0].data.cpu().numpy(),
                                                   (h[i][0].data.cpu().numpy(),
                                                    v[i][0].data.cpu().numpy(),
                                                    d[i][0].data.cpu().numpy())), 'haar')
                            '''
                        loss1 = my_mse_loss(Co*255, Ca*255)  #
                        loss2 = my_mse_loss(im1*255, watermark*255)


                        #watermark = torch.from_numpy(watermark).to(device)

                        # loss2 = my_mse_loss(watermark*255, torch.from_numpy(np.repeat(im[np.newaxis, :, :], inputs.shape[0], axis=0)).to(device)*255)*inputs.shape[0] # 计算loss
                        # loss2.requires_grad=True
                        loss = (loss1) * u + (loss2) * (1 - u)
                        if phase == 'train':
                            # loss1.backward(retain_graph=True)#)
                            loss.backward()  # 最后一次不加retain_graph=True)#
                            optimizer1.step()
                            optimizer2.step()
                        PrintF(index * batch_size + inputs.shape[0], int(dataset_sizes[phase]), phase)
                    # 记录 loss 和 准确率
                    running_loss[index] = loss.item()
                    running_loss1[index] = loss1.item()
                    running_loss2[index] = loss2.item()
            epoch_loss = torch.mean(running_loss)
            epoch_loss1 = torch.mean(running_loss1)
            epoch_loss2 = torch.mean(running_loss2)
            if epoch_loss < min_loss:
                torch.save(resNet.state_dict(), model_path + 'best_resNet-4c.pkl')
                torch.save(rresNet.state_dict(), model_path + 'best_rresNet-4c.pkl')
                min_loss = epoch_loss
            print('\nLoss: {:.4f} \t Loss1: {:.4f} \t Loss2: {:.4f} \t Time: {:.4f} sec'.
                  format(epoch_loss.item(), epoch_loss1.item(), epoch_loss2.item(), time.time() - start))
    print("Training Finshed!!!    Total Time Cost：{:.4f}second!!!".format(time.time() - time_start))
    torch.save(resNet.state_dict(), model_path + 'resNet.pkl')
    torch.save(rresNet.state_dict(), model_path + 'rresNet.pkl')
    imshow(Co[0].detach().cpu())
    imshow(Ca[0].detach().cpu())
    imshow(watermark[0].detach().cpu())
    imshow(im1[0].detach().cpu())
    print(psnr1(Ca[0].detach().cpu() * 255, Co[0].detach().cpu() * 255))
    print(psnr1(watermark[0].detach().cpu() * 255, im * 255))
    print(ssim(Ca[0].detach().cpu() * 255, Co[0].detach().cpu() * 255))
    print(ssim(watermark[0].detach().cpu() * 255, im * 255))
    '''
    im2=plt.imread(r'F:\DeepLearning\data\VOC2007\JPEGImages\test/009961.jpg')
    aaa=transforms.ToTensor()(im2)
    aaa=transforms.Resize((256, 256))(aa
                    running_psnr1 = running_psnr1 + psnr01.item()
                    running_psnr2 = running_psnr2 + psnr02.item()a)
    Ca=(aaa[np.newaxis, :, :, :]).to(device)

    plt.figure("lena")
    arr=watermark[0].detach().cpu().flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='gray', alpha=0.75)  
    plt.show()
    '''