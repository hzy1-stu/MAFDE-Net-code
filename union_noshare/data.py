import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from my_math import complex2real, real2complex, fft2c, ifft2c, sos
import scipy.io as io
from tqdm import tqdm
from torchvision.transforms import transforms


def normalization(data, dtype='max'):
    if dtype == 'max':
        norm = np.max(np.abs(data))
    elif dtype == 'no':
        norm = 1.0
    else:
        raise ValueError("Normalization has to be in ['max', 'no']")

    data = data / norm.astype(float)
    return data


class DataPrepare(Dataset):
    def __init__(self, use, maskname, root='./dataset/'):
    # ///////////////////////////////////////////此处是加载数据前的准备工作/////////////////////////////////////////////////////////
        datadir = root + 'data'
        maskdir = root + 'mask'
        FileList = []
        use_label = use + '_label'
        # os.walk会遍历我们指定的文件夹
        #  每一层遍历：
        #   root：保存的是当前遍历文件夹的绝对路径
        #   dirs：保存当前文件夹下的所有子文件夹的名称
        #   files：保存当前文件夹下的所有文件的名称
        for root, dirs, files in os.walk(datadir):
            for file in files:
                FileList.append(file)
        # 从FileList中找到train.h5文件，用于训练
        for i in FileList:
            if i == use + '.h5':
                filename = i
        for j in FileList:
            if j == use_label + '.h5':
                filelabel = j


        # total表示数据预处理的进度总量，bar的值例如：     0%|     |0/4[00:00<...]
        bar = tqdm(total=4)
        # preparing data
        # set_description放置在进度去前面，是对于进度条的描述：例如：preparing dataset for train: 0%|  |0/4[00:00<...](use的值为字符串train)
        bar.set_description('Preparing dataset for %s' % use)

        "data为实数，此时shape为(500, 256, 256, 4)"
        data = h5py.File(os.path.join(datadir, filename),'r')[use][:]    # ground-truth multichannel(多通道原图)(复数形式)
        data = normalization(data, dtype='max')
        "data_label为实数，此时shape为(500, 256, 256, 3)"
        data_label = h5py.File(os.path.join(datadir, filelabel),'r')[use_label][:]
        data_label = np.transpose(data_label, (0, 3, 1, 2))


        data = data[:, :, :, 1:2]
        data_label = data_label[:, 1:2, :, :]

        "data进行维度顺序交换，此时shape为(500, 1, 256, 256)"
        data = np.transpose(data, (0, 3, 1, 2))
        "data从实数转化为复数，此时shape为(500, 1, 256, 256)"
        data = data.astype(np.complex64)
        "kdata的shape为(500, 1, 256, 256)"
        kdata = fft2c(data)    # full-sample k-space data(复数形式)
        # 此处更新进度条迭代值，在这里从0/4变成1/4------
        bar.update(1)
        # 2.生成掩码Z
        # generate mask
        # 很多数据集都是mat格式的标注信息，使用模块scipy.io的函数loadmat和savemat可以实现Python对mat数据的读写。
        # Python使用Scipy库中的io.loadmat读取.mat文件，并获取数据部分
        # scipy.io.loadmat(file_name, mdict=None, appendmat=True, **kwargs)，其中file_name为mat文件所在路径
        # 注意：函数返回值为字典类型dict。之后还需要通过后续操作将值提取出来（字典操作来提取键值对的值）
        #      关于为什么scipy.io.loadmat("FilePath")的结果是一个字典，这是因为一个.mat文件中存在多个变量，每一个变量名都对应相应的数据，也就是变量名和变量值的键值对。
        # 在这里字典通过键['mask']来提取掩码的取值
        "读出来mask的shape为（256，256）"
        mask = io.loadmat(os.path.join(maskdir, maskname))['mask']
        # np.tile函数是将数组进行复制，其中256和256各复制一次（等于没动）；通道数=1复制2次，变成2个通道；对于最后的1（可以reshape出这个1），重复10次，变成数量10
        "这一步就是要使mask的shape跟数据data的大小一样,即shape为(256, 256, 1, 500)"
        mask = np.tile(mask.reshape((256, 256, 1, 1)), (1, 1, 1, data.shape[0]))
        "将mask的shape中的维度交换位置，确保每个维度跟data相照应,即shape为(500, 1, 256, 256)"
        mask = np.transpose(mask, (3, 2, 0, 1))
        # 此处更新进度条迭代值，在这里从1/4变成2/4
        bar.update(1)
        # 3.Y进行欠采样
        # under-sample
        u_kdata = np.zeros(kdata.shape, dtype=np.complex64)
        # 用生成的掩码与K空间数据进行逐元素相乘，进行欠采样操作，得到欠采样K空间数据
        # mask就是1和0，可以和复数直接乘的
        "欠采样K空间数据u_kdata的shape为(500, 1, 256, 256)"
        u_kdata = mask * kdata
        # 将欠采样K空间数据进行逆傅里叶变换得到欠采样图像数据（零填充图像）
        "零填充图像u_data的shape为(500, 1, 256, 256)"
        u_data = ifft2c(u_kdata)
        # 此处更新进度条迭代值，在这里从2/4变成3/4
        bar.update(1)
        # Separating real and imaginary parts and convert to tensor
        # 4.对变量进行初始化操作，以便这个类中的方法进行调用这些变量
        #   将数据的实部和虚部进行分离并将其转换成tensor数据格式（这里所提到的实部和虚部分离，其实就是将复数转换为实数的过程）
        # 磁共振图像都是复数，跟K空间数据格式是一样的。另外网络只能处理实数，不能处理复数。所以在这里准备网络的输入数据时，要切记将数据的复数形式转换为实数，再传进网络进行处理
        "shape为(500, 2, 256, 256)"
        self.u_data = complex2real(torch.from_numpy(u_data))
        "shape为(500, 2, 256, 256)"
        self.u_kdata = complex2real(torch.from_numpy(u_kdata))
        "shape为(500, 256, 256)"
        self.data = sos(torch.from_numpy(data))
        "shape为(500, 256, 256)"
        self.data_label = sos(torch.from_numpy(data_label))
        "shape为(500, 1, 256, 256)"
        self.mask = torch.from_numpy(mask)
        # 此处更新进度条迭代值，在这里从3/4变成4/4
        bar.update(1)
        bar.close()

        self.DataLen = self.data.shape[0]

    def __len__(self):
        return self.DataLen

    def __getitem__(self, idx):
        return self.u_data[idx], self.mask[idx], self.data[idx], self.u_kdata[idx], self.data_label[idx]


# 画图只能画实数，复数是不能画出来的。在前面已经将网络的输入数据转换成实数了，若没有，则在这里需要对复数取模才能画图
def psnr_plot(item, path, name):
    figname = os.path.join(path, name)
    x = range(len(item))
    # plt.figure使在plt中绘制一张图片（而subplot是创建单个子图，其他的详见csdn收藏夹）
    # （1）是图像编号
    plt.figure(1)
    # x轴数据是range(epoch)，y轴数据是存储的PSNR值的列表，label指定线条的标签为PSNR，‘b’：蓝色
    plt.plot(x, item, 'b', label='PSNR')
    # 不显示网格线
    plt.grid(False)
    # x轴标签
    plt.xlabel('Epoch')
    # y轴标签
    plt.ylabel('PSNR')
    # plt.legend()的作用：在plt.plot() 定义后plt.legend() 会显示该 plot中 label 的内容，
    # plt.legend(loc)是设置图例的位置，而best代表0，意思是自适应
    plt.legend(loc='best')
    # 保存图片到指定路径中
    plt.savefig(figname)
    plt.show()


def ssim_plot(item, path, name):
    figname = os.path.join(path, name)
    x = range(len(item))
    plt.figure(1)
    plt.plot(x, item, 'c', label='SSIM')
    plt.grid(False)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend(loc='best')
    plt.savefig(figname)
    plt.show()

def dice_plot(item, path, name):
    figname = os.path.join(path, name)
    x = range(len(item))
    # plt.figure使在plt中绘制一张图片（而subplot是创建单个子图，其他的详见csdn收藏夹）
    # （1）是图像编号
    plt.figure(1)
    # x轴数据是range(epoch)，y轴数据是存储的PSNR值的列表，label指定线条的标签为PSNR，‘b’：蓝色
    plt.plot(x, item, 'b', label='DICE')
    # 不显示网格线
    plt.grid(False)
    # x轴标签
    plt.xlabel('Epoch')
    # y轴标签
    plt.ylabel('DICE')
    # plt.legend()的作用：在plt.plot() 定义后plt.legend() 会显示该 plot中 label 的内容，
    # plt.legend(loc)是设置图例的位置，而best代表0，意思是自适应
    plt.legend(loc='best')
    # 保存图片到指定路径中
    plt.savefig(figname)
    plt.show()

def loss_plot(item1, item2, path, name):
    figname = os.path.join(path, name)
    plt.figure(1)
    plt.plot(range(len(item1)), item1, 'r', label='Train Loss')
    plt.plot(range(len(item2)), item2, 'g', label='Validate Loss')
    plt.grid(False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(figname)
    plt.show()


def savefig(image, path, batch, type):
    image = image.cpu().numpy()
    figurepath = os.path.join(path, 'figure/')
    if not os.path.exists(figurepath):
        os.makedirs(figurepath)
    figname = os.path.join(figurepath, 'No.%d-%s.png' % (batch+1, type))
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.savefig(figname)

def save_result(raw, under, rec, seg, label, save_path, i):
    # save figure
    savefig(raw, save_path, i, 'groundtruth')
    savefig(under, save_path, i, 'undersampling')
    savefig(rec, save_path, i, 'reconstructed')
    savefig(seg, save_path, i, 'segmention')
    savefig(label, save_path, i, 'label')
    # save mat
    mat_path = os.path.join(save_path, 'mat/')
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)

    # 对mat文件进行保存（先输入文件所在路径，然后将需要保存的变量以字典的形式传进去）
    io.savemat(os.path.join(mat_path, 'No.%d-%s.mat' % (i + 1, 'groundtruth')), {'data': raw.cpu().numpy()})
    io.savemat(os.path.join(mat_path, 'No.%d-%s.mat' % (i + 1, 'undersampling')), {'data': under.cpu().numpy()})
    io.savemat(os.path.join(mat_path, 'No.%d-%s.mat' % (i + 1, 'reconstructed')), {'data': rec.cpu().numpy()})
    io.savemat(os.path.join(mat_path, 'No.%d-%s.mat' % (i + 1, 'segmention')), {'data': seg.cpu().numpy()})
    io.savemat(os.path.join(mat_path, 'No.%d-%s.mat' % (i + 1, 'label')), {'data': label.cpu().numpy()})
