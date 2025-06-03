import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from my_math import get_rmse, get_psnr, get_ssim, complex2real, real2complex, sos, torch_fft2c
import shutil
import argparse
from networks import MICCANlong
from utils import *
import random
import scipy.io as io
from data import DataPrepare, savefig, psnr_plot, dice_plot, loss_plot, ssim_plot, save_result
from time import time
from tqdm import tqdm

# 定义函数，保存最新和最佳的模型
# checkpoint用来保存每次训练好的模型参数(也可以理解为保存最新的模型)，而model_best是从保存好的模型参数中找出最佳模型并保存
def save_checkpoint(state, save_path, is_best, filename='checkpoint.pth.tar'):
    # 把序列化的对象保存到硬盘。它利用了 Python 的 pickle 来实现序列化。模型、张量以及字典都可以用该函数进行保存；
    torch.save(state, os.path.join(save_path,filename))

    if is_best:
        # shutil.copyfile(file1,file2)
        # file1为需要复制的源文件的文件路径,file2为目标文件的文件路径+文件名.
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))

def validate(val_loader, model, cri, epoch, device, test=False):
    global save_path
    # 运用AverageMeter函数计算每一个epoch的平均损失，即将每个样本的损失取一下平均。
    # 调用此函数的时候，要先声明一下
    val_loss = AverageMeter()
    # model.eval()的作用是 不启用 Batch Normalization 和 Dropout。
    # 如果模型中有 BN 层（Batch Normalization）和 Dropout，在 测试时 添加 model.eval()。
    # model.eval() 是保证 BN 层能够用 全部训练数据 的均值和方差，即测试过程中要保证 BN 层的均值和方差不变。对于 Dropout，model.eval() 是利用到了 所有 网络连接，即不进行随机舍弃神经元。
    model.eval()

    allpsnr = []
    allssim = []
    alldice = []
    allIOU = []

    i = 0

    bar = 'Test' if test else 'Validate'

    # loss_mse = nn.MSELoss()

    "定义分割损失"
    dice_loss = DiceLoss(2)
    dice_ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for u_data, mask, data, u_kdata, data_label in tqdm(val_loader, desc='%s  |  Epoch-' % bar + str(epoch+1)):
            u_data = u_data.float().to(device)
            mask = mask.float().to(device)
            data = data.float().to(device)
            u_kdata = u_kdata.float().to(device)
            data_label = data_label.float().to(device)

            data_label = torch.clamp(data_label, 0, 1)

            # model forward
            "图像域重建结果"
            re = model(u_data, u_kdata, mask)
            "K空间域重建结果"
            k_re = complex2real(torch_fft2c(real2complex(re)))
            "将双域重建结果放入分割网络得到分割结果"
            reconsimage = model(re, k_re, mask, False)
            "将重建结果转为复数并进行sos运算，以便后续计算指标和保存数据"
            re = sos(real2complex(re))
            # 计算损失值
            "重建损失"
            loss_rec = cri(re, data)

            "分割损失"
            loss_ce = dice_ce(reconsimage, data_label.long())
            loss_dice = dice_loss(reconsimage, data_label, softmax=True)
            loss_seg = loss_dice

            "总损失"
            loss = loss_rec + loss_seg

            "计算dice指标及报存分割图"
            dice_seg = dice_coef2(torch.argmax(torch.softmax(reconsimage, dim=1), dim=1).cpu().numpy(),
                                  data_label.cpu().numpy())
            IOU = sespiou_coefficient2(torch.argmax(torch.softmax(reconsimage, dim=1), dim=1).cpu().numpy(),
                                       data_label.cpu().numpy())

            reconsimage = torch.argmax(torch.softmax(reconsimage, dim=1), dim=1)

            "保存数据"
            seg = torch.squeeze(reconsimage)
            rec = torch.squeeze(re)
            raw = torch.squeeze(data)
            under = torch.squeeze(sos(real2complex(u_data)))
            label = torch.squeeze(data_label)

            # 计算相应的评估值
            psnr = get_psnr(rec, raw)
            ssim = get_ssim(rec, raw)
            allpsnr.append(psnr)
            allssim.append(ssim.item())
            alldice.append(dice_seg)
            allIOU.append(IOU)
            # 保存测试的结果
            # save result when testing
            if test:
                save_result(raw, under, rec, seg, label, save_path, i)

            # 记录验证的损失
            # AverageMeter是根据batch_size的大小来计算损失，所以括号里面会添加u_data.size(0)
            # record validation loss
            val_loss.update(loss.item(), u_data.size(0))

            i = i+1
        # 记录测试过程中每一批次的评分到txt文件中
        # record score of every batch in txt while testing
        # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会
        if test:
            # np.savetxt(保存路径，保存数据，使用默认分割符（空格）并保留四位小数)
            np.savetxt(os.path.join(save_path, 'PSNR_batches_test.txt'), np.asarray(allpsnr), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'SSIM_batches_test.txt'), np.asarray(allssim), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'dice_batches_test.txt'), np.asarray(alldice), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'IOU_batches_test.txt'), np.asarray(allIOU), fmt='%.4f')

        # print out average scores
        # 输出质量评估分数的平均值
        avgpsnr = np.mean(np.asarray(allpsnr))
        avgssim = np.mean(np.asarray(allssim))
        avgdice = np.mean(np.asarray(alldice))
        avgIOU = np.mean(np.asarray(allIOU))

        print(' * Epoch-'+str(epoch+1)+'\tAverage Validation Loss {:.9f}'.format(val_loss.avg))
        print(' * Epoch-'+str(epoch+1)+'\tAverage PSNR {:.4f}'.format(avgpsnr))
        print(' * Epoch-'+str(epoch+1)+'\tAverage SSIM {:.4f}'.format(avgssim))
        print(' * Epoch-' + str(epoch + 1) + '\tAverage dice {:.4f}'.format(avgdice))
        print(' * Epoch-' + str(epoch + 1) + '\tAverage iou {:.4f}'.format(avgIOU))

        return val_loss.avg, avgpsnr, avgssim, avgdice, avgIOU


# data数量为10，每个batch_size大小为1，所以总共有10个batch。epoch默认值设置为100，所以相当于内循环里面把每一个batch拿出来训练，而外循环里面把内循环循环100次
parser = argparse.ArgumentParser(description='Main function arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', default=False, help='train or test')
parser.add_argument('--multi', default=False, help='train on multi gpu or not')
parser.add_argument('--seed', default=6, type=int, help='random seed')
parser.add_argument('--loss', default='l1', type=str, help='loss function')
parser.add_argument('--nblock', default=1, type=int, help='number of block')
parser.add_argument('--gpuid', default='0', type=str, help='gpu id')  # 这里的默认值0指的是GPU的编号
parser.add_argument('--bs', default=6, type=int, help='batchsize')
parser.add_argument('--epoch', default=400, type=int, help='number of epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--savepath', default='result', type=str, help='save file path')
parser.add_argument('--maskname', default='mask_vddisc_256x256_R3', type=str, help='mask name')


# global variables
"""全局变量"""
n_iter = 0
best_loss = -1

"""主函数（Pytroch利用GPU训练模型需要将设计好的模型和数据放入指定的GPU上，至于损失函数个人认为不需要放到GPU上，当然放上去也不会报错）"""
# main function
def main():

    testloder = DataLoader(DataPrepare('test', args.maskname), batch_size=1, shuffle=False)
    # 在Pytorch中构建好一个模型后，一般需要进行预训练权重中加载。torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    # ['state_dict']意思是加载保存在state_dict里面的参数
    network.load_state_dict(torch.load(args.savepath + '/model_best.pth.tar')['state_dict'])
    validate(testloder, network, loss, 0, device, test=True)
    print('Finish   |   Consume:{:.2f}min'.format((time() - time0) / 60))



if __name__ == '__main__':
    main()