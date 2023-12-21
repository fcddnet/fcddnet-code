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
# root = r"F:\DeepLearning\data\VOC+NET/"
# root512 = r"F:\DeepLearning\data\VOC+NET\Test512"
root = r"F:\DeepLearning\data\VOC+NET/test2"
root512 = r"F:\DeepLearning\data\VOC+NET\test2"

model_path1 = r"./data\best_FCdDNet-5m0.5.pkl"
model_path2 = r"./data\best_rFCdDNet-5m0.5.pkl"#4c
batch_size = 1

def imsave1(inp, path):
    """Imshow for Tensor."""
    # 逆转操作，从 tensor 变回 numpy 数组需要转换通道位置
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imsave(r"C:\Users\Doc_m\Desktop/" + path, inp)
# 数据预览
def imshow(inp):  # 若归一化，则要修改方差和均值
    """Imshow for Tensor."""
    # 逆转操作，从 tensor 变回 numpy 数组需要转换通道位置
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
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
    if imageA.shape[0] == 3:
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

if __name__ == "__main__":
    #coeffs = pywt.dwt2(im, 'haar')
    # -----------------ready the dataset--------------------------
    # 数据读取，将原始数据随机划分为三种数据训练集、验证集合测试集；前两种是用于模型的训练和调整，最后种是用于模型最终的评估
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    data_transforms1 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    all_dataset = datasets.ImageFolder(root, transform=data_transforms)
    all_secret = datasets.ImageFolder(root512, transform=data_transforms1)
    all_dataset_lenth = len(all_dataset)  # 获取所有数据长度
    all_secret_lenth = len(all_secret)  # 获取所有数据长度

    test,_ = torch.utils.data.random_split(dataset=all_dataset,
                                                       lengths=[all_dataset_lenth,0])
    test_secret,_ = torch.utils.data.random_split(dataset=all_secret,
                                                       lengths=[all_secret_lenth,0])
    # train, test, valid = torch.utils.data.random_split(dataset=all_dataset,
    #                                                    lengths=[int(all_dataset_lenth * 0.8),
    #                                                             int(all_dataset_lenth * 0.15),
    #                                                             all_dataset_lenth - int(all_dataset_lenth * 0.8) - int(
    #                                                                 all_dataset_lenth * 0.15)])
    # train_secret, test_secret, valid_secret = torch.utils.data.random_split(dataset=all_secret,
    #                                                    lengths=[int(all_secret_lenth * 0.8),
    #                                                             int(all_secret_lenth * 0.15),
    #                                                             all_secret_lenth - int(all_secret_lenth * 0.8) - int(
    #                                                                 all_secret_lenth * 0.15)])
    # len(valid)+len(train)+len(test) == len(all_dataset)#验证一下数据长度是否一致
    # 数据加载
    #trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    #validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
    #trainloader_secret = torch.utils.data.DataLoader(train_secret, batch_size=batch_size, shuffle=True)
    testloader_secret = torch.utils.data.DataLoader(test_secret, batch_size=batch_size, shuffle=True)
    #validloader_secret = torch.utils.data.DataLoader(valid_secret, batch_size=batch_size, shuffle=True)

    ocdh = FCdDNet()
    print(ocdh)
    rocdh= ReFCdDNet()
    print(rocdh)

    # 合并便于训练与评估，设置参数
    dataloaders = {
        # 'train': trainloader,
        # 'val': validloader,
        'test': testloader
    }
    dataloaders_secret = {
        # 'train': trainloader_secret,
        # 'val': validloader_secret,
        'test': testloader_secret
    }
    dataset_sizes = { 'test': len(test)}#'train': len(train), 'val': len(valid),

    # for inputs, labels in dataloaders['train']:
    #     print(inputs.shape)
    ocdh = ocdh.to(device)
    rocdh = rocdh.to(device)
    ocdh.load_state_dict(torch.load(model_path1))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
    rocdh.load_state_dict(torch.load(model_path2))  # 'best_gansG.pkl'))#'gansG1-n-1.pkl'))#
       # 开始训练
    print("Training on ", device)
    time_start = time.time()
    # 训练网络
    # 迭代epoch
    min_loss = torch.Tensor([99999]).to(device)
    psnr_max1=0
    psnr_max2=0
    for phase in ['test']:
        # 注意训练和验证阶段，需要分别对 model 的设置
        ocdh.train(False)  # 评估模式会关闭Dropout
        rocdh.train(False)  # 评估模式会关闭Dropout
        start = time.time()
        psnr01 = torch.zeros(dataset_sizes[phase]).to(device)
        psnr02 = torch.zeros(dataset_sizes[phase]).to(device)
        ssim01 = torch.zeros(dataset_sizes[phase]).to(device)
        ssim02 = torch.zeros(dataset_sizes[phase]).to(device)
        sums = torch.zeros(dataset_sizes[phase]).to(device)
        with torch.no_grad():
            for index, (inputs, _) in enumerate(dataloaders[phase]):
                im0, _ = iter(testloader_secret).next()
                im1 = im0[:inputs.shape[0],:,:,:].to(device)
                Co = inputs.to(device)
                # 只有训练阶段才追踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    Ca = ocdh(Co, im1)
                    watermark = rocdh(Ca)
                    Ca[Ca<=0]=0
                    Ca[Ca>=1]=1
                    watermark[watermark<=0]=0
                    watermark[watermark>=1]=1
                    psnr01[index] = psnr1(Co * 255, Ca * 255)
                    psnr02[index] = psnr1(watermark * 255,
                                            im1[0] * 255)
                    ssim01[index] = ssim(Co[0].detach().cpu() * 255, Ca[0].detach().cpu() * 255)
                    ssim02[index] = ssim(watermark[0,:,:,:].detach().cpu() * 255, im1[0,:,:,:].detach().cpu() * 255)
                    #ssim02 = ssim02 + ssim()
                # 记录 loss 和 准确率
                if psnr01[index]>psnr_max1:
                    psnr_max1=psnr01[index]
                if psnr02[index]>psnr_max2:
                    psnr_max2=psnr02[index]
                if psnr01[index] < 32:
                    #imshow(Co[0].detach().cpu())
                    print(psnr01[index])
                if psnr02[index] < 24:
                    print(psnr02[index])


                imshow(Co[0].detach().cpu())
                imshow(watermark[0].detach().cpu())
                abss=abs(Co[0].detach().cpu()*255-Ca[0].detach().cpu()*255)
                sums[index] = sum(sum(sum(abss)))
                #print(sums[index])
                PrintF(index * batch_size + inputs.shape[0], int(dataset_sizes[phase]), phase)

        if device.type == 'cuda':
            psnr01 = psnr01.cpu()
            psnr02 = psnr02.cpu()
            ssim01 = ssim01.cpu()
            ssim02 = ssim02.cpu()
        epoch_psnr1 = torch.mean(psnr01)
        epoch_psnr2 = torch.mean(psnr02)
        epoch_ssim1 = torch.mean(ssim01)
        epoch_ssim2 = torch.mean(ssim02)
        sums1 = torch.mean(sums)
        print('\nPSNR1: {:.4f} \t PSNR2: {:.4f} \nSSIM1: {:.4f} \t '
              'SSIM2: {:.4f} \t Time: {:.4f} sec'.
              format(epoch_psnr1.item(), epoch_psnr2.item(), epoch_ssim1.item(), epoch_ssim2.item(),
                     time.time() - start))
    print("Training Finshed!!!    Total Time Cost：{:.4f}second!!!".format(time.time() - time_start))



    imsave1(Co[0].detach().cpu(), 'image8_c1.png')
    imsave1(Ca[0].detach().cpu(), 'image8_c2.png')
    imsave1(im1[0].detach().cpu(), 'image8_c3.png')
    imsave1(watermark[0].detach().cpu(), 'image8_c4.png')

    error=Ca[0].detach().cpu()-Co[0].detach().cpu()
    imshow(abs(error*20))

    r, g, b = Co[0].detach().cpu()*255
    ar = np.array(r).flatten()
    plt.hist(ar, bins=256, normed=1, facecolor='r', edgecolor='r', alpha=0.75)
    ag = np.array(g).flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g', edgecolor='g', alpha=0.75)
    ab = np.array(b).flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.show()
    r, g, b = Ca[0].detach().cpu()*255
    ar = np.array(r).flatten()
    plt.hist(ar, bins=256, normed=1, facecolor='r', edgecolor='r', alpha=0.75)
    ag = np.array(g).flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g', edgecolor='g', alpha=0.75)
    ab = np.array(b).flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.show()

    r, g, b = im1[0].detach().cpu()*255
    ar = np.array(r).flatten()
    plt.hist(ar, bins=256, normed=1, facecolor='r', edgecolor='r', alpha=0.75)
    ag = np.array(g).flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g', edgecolor='g', alpha=0.75)
    ab = np.array(b).flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.show()
    r, g, b = watermark[0].detach().cpu()*255
    ar = np.array(r).flatten()
    plt.hist(ar, bins=256, normed=1, facecolor='r', edgecolor='r', alpha=0.75)
    ag = np.array(g).flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g', edgecolor='g', alpha=0.75)
    ab = np.array(b).flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.show()



    imshow(Co[0].detach().cpu())
    imshow(watermark[0].detach().cpu())
    imshow(Ca[0].detach().cpu())
    print(psnr1(Ca[0].detach().cpu() * 255, Co[0].detach().cpu() * 255))
    print(psnr1(watermark[0].detach().cpu() * 255, im * 255))
    print(ssim(Ca[0].detach().cpu() * 255, Co[0].detach().cpu() * 255))
    print(ssim(watermark[0].detach().cpu() * 255, im * 255))
    imshow((Co[0].detach().cpu()-Ca[0].detach().cpu())*20)
