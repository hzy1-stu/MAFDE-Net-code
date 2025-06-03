import torch.nn as nn
import torch
from torch.nn.init import normal_
from my_math import real2complex, complex2real, torch_fft2c, torch_ifft2c, sos
import math
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import h5py
from network.Res2Net import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from unet import unet
from unet1 import unet1


# i_DC_layer
class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()

    # 因为DC层是接在UCA网络后面，从UCA网络出来的是实数(网络只能处理实数)，所以在DC层中要进行转复数操作（但切记磁共振图像和K空间数据都是复数）
    def forward(self, mask, x_rec, xk_under):
        # x_rec = 从网络里出来的重建过的图像 + 零填充图像(输入)
        # xk_under为欠采样k空间数据
        # 1.将图像转换成复数，是需要在傅里叶变换之前需要做的。比如在DC层，它要提前转换成复数，然后进行傅里叶变换转换成K空间数据，进行数据保真运算
        x_rec_r2c = real2complex(x_rec)
        # 2.进行傅里叶变换，变成K空间数据
        # 注意：根据傅里叶性质，输入是实数或者复数，输出是复数。但一般处理图像的时候都是先将其转换为复数，再进行傅里叶变换。因为如果直接传进去，会造成错误
        x_rec_fft = torch_fft2c(x_rec_r2c)
        # 3.这里再将复数转换成实数是因为mask和xk_under是实数，所以转换成实数才能进行保真运算、
        # 这里的xk_under实部跟虚部是分开运算的（具体详见数据处理：xk_under复数转为实数）
        x_rec_c2r = complex2real(x_rec_fft)


        # 1.这里进行拼接，是因为单个mask是15个通道，而数据是30个通道的，所以这里要拼接两个mask(拼接到第一个维度上)
        masks = torch.cat([mask, mask], 1)
        # 2.进行数据的保真运算
        # 这里的xk_under实部跟虚部是分开运算的（具体详见数据处理：xk_under复数转为实数）
        output = (1.0 - masks) * x_rec_c2r + xk_under


        # 1.将输出结果转换为复数形式(这里也是先将数据转换为复数，再进行逆傅里叶变换)
        output_r2c = real2complex(output)
        # 2.对结果进行傅里叶逆变换，将频域数据转换为时域数据
        output_ifft = torch_ifft2c(output_r2c)
        # 3.将时域数据转换为实数，便于进入下一层神经网络进行处理
        output_c2r = complex2real(output_ifft)

        return output_c2r


# 总的网络结构
# MICCAN with long residual, reconstruction block could be cascaded blocks, UNet, UCA(UNet with attention)
class MICCANlong(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer):
        super(MICCANlong, self).__init__()

        # 这两步主要是初始化定义所需要的网络类型
        self.layer = unet(in_channel, out_channel) # 里面的列表生成式表示有两个UNetCSE
        self.klayer = unet1(in_channel, out_channel)
        self.dc = DC_layer()

        # 其中n_layer是级联的层数，代表有两个UCA和DC层
        self.nlayer = n_layer

    def forward(self, x_under, xk_under, mask, rec=True):
        x_rec_dc = x_under

        # 重建前向传播
        if rec:
            x_rec = self.layer(x_rec_dc, xk_under, mask)

            return x_rec

        # 分割前向传播
        else:
            x_rec = self.klayer(x_rec_dc)

            return x_rec