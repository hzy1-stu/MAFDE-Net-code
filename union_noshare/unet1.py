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


# 总的网络结构
# MICCAN with long residual, reconstruction block could be cascaded blocks, UNet, UCA(UNet with attention)
class MICCANlong(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer):
        super(MICCANlong, self).__init__()

        # 这两步主要是初始化定义所需要的网络类型
        self.layer = nn.ModuleList([UNetCSE(in_channel, out_channel) for _ in range(n_layer)]) # 里面的列表生成式表示有两个UNetCSE
        self.nlayer = n_layer

    def forward(self, x):
        # x_under是欠采样时域数据(相当于零填充图像)
        x_rec_dc = x
        for i in range(self.nlayer):  # range(1)

            x_rec_dc = self.layer[i](x_rec_dc)

        return x_rec_dc

def unet1(in_channel, out_channel):

    model = MICCANlong(2, 2, 1)
    checkpoint = torch.load(
        '/home/huangzhenyu/new_four/CHAOS/union_noshare/seg_model_best.pth.tar')[
        'state_dict']
    model.load_state_dict(checkpoint)
    return model


class trans(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1, dropout=False):
        super(trans, self).__init__()
        self.conv_dim = in_ch // 2
        self.trans_dim = in_ch // 2
        self.head_dim = 32
        self.window_size = 8
        self.drop_path = 0.0
        self.type = 'SW'

        assert self.type in ['W', 'SW']

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x1 = x + res

        return x1


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1, dropout=False):
        super(double_conv_up, self).__init__()
        self.drop = 0.
        self.ConvV2 = ConvV2_concat_Block(in_ch)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.ConvV2(x)
        x = self.conv_3(x)
        x3 = x + x1
        return x3


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(inconv, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding),
            nn.ReLU(inplace=True),  # 设置成TRUE这样做的好处就是能够节省运算内存，不用多存储额外的变量
            nn.Conv2d(out_ch, out_ch, 3, padding=padding),
            nn.ReLU(inplace=True),  # 设置成TRUE这样做的好处就是能够节省运算内存，不用多存储额外的变量
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Conv2d(in_ch, 64, 3, padding=padding)

    def forward(self, x):
        x1 = self.conv_2(self.conv_1(x))
        x2 = x1 + self.conv_3(x)
        return x2

class up(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                # 用双线插值对特征图放大两倍，使其跟拼接过来的浅层特征图大小保持一致
                # scale_factor指定输出大小为输入的多少倍数;mode:可使用的上采样算法;align_corners为True，输入的角像素将与输出张量对齐
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 使用卷积层将其通道数减半（加入padding来保证特征图不变），利于拼接
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x1, x2):
        # 上采样的过程中需要对两个特征图进行融合，通道数一样并且尺寸也应该一样，x1是上采样获得的特征，而x2是下采样获得的特征，
        # 首先对x1进行反卷积使其大小变为输入时的2倍，首先需要计算两张图长宽的差值，作为填补padding的依据，由于此时图片的表示为（C,H,W）
        # 因此diffY对应的图片的高，diffX对应图片的宽度， F.pad指的是（左填充，右填充，上填充，下填充），其数值代表填充次数，因此需要/2，最后进行融合剪裁

        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        # Math.ceil()  “向上取整”， 即小数部分直接舍去，并向正数部分进1
        x1 = F.pad(x1, (math.ceil(diffY / 2), int(diffY / 2),
                        math.ceil(diffX / 2), int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
                # 最大池化只改变图片尺寸的大小，不改变通道数。参数stride默认值为kernel大小
                nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)


    def forward(self, x):
        x = self.conv(x)
        return x


# UNet with channel-wise attention, input arguments of CSE_block should change according to image size
class UNetCSE(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetCSE, self).__init__()
        self.conv_layer_in = nn.Conv2d(n_channels, 3, kernel_size=1, stride=1, bias=False)
        self.conv_layer = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.inc = inconv(n_channels, 64, 1)
        self.inc_res1 = nn.Conv2d(256, 64, 1)
        self.inc_res2 = nn.Conv2d(512, 64, 1)
        self.inc_res3 = nn.Conv2d(1024, 64, 1)
        self.inc_res4 = nn.Conv2d(2048, 64, 1)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.trans1 = trans(64, 64, 1)
        self.trans2 = trans(64, 64, 1)
        self.trans3 = trans(64, 64, 1)
        self.trans4 = trans(64, 64, 1)

        self.up3 = up(64, 64, 1)

        self.up2 = up(64, 64, 1)

        self.up1 = up(64, 64, 1)

        self.outc = outconv(64, n_classes)

        self.down = down(64, 64, 1)

        self.head = _DAHead(128, 64, aux=False)

        self.invert1 = InvertedResidual(64, 64, 1, 1)
        self.invert2 = InvertedResidual(64, 64, 1, 6)
        self.invert3 = InvertedResidual(64, 64, 1, 6)
        self.invert4 = InvertedResidual(64, 64, 1, 6)

        self.scope = DySample(64)
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.down(x1)

        "SW-MSA编码器"
        x2 = self.trans1(x1)
        x3 = self.down(x2)
        x4 = self.trans2(x3)
        x5 = self.down(x4)
        x6 = self.trans3(x5)
        x7 = self.down(x6)
        x8 = self.trans4(x7)

        "Res2Net编码器"
        res1, res2, res3, res4 = self.resnet(self.conv_layer_in(x))
        res1 = self.inc_res1(res1)  # 1,64,128,128
        res2 = self.inc_res2(res2)  # 1,64,64,64
        res3 = self.inc_res3(res3)  # 1,64,32,32
        res4 = self.inc_res4(res4)  # 1,64,16,16

        "两个编码器最后输出进行交互"
        x8 = F.interpolate(x8, res4.size()[2:], mode='bilinear', align_corners=False)
        dual_attention = self.head(torch.cat([x8, res4], dim=1))   # 1,64,16,16

        "解码器"
        decode_1 = torch.cat([dual_attention, res4], dim=1)
        decode_2 = self.invert1(self.conv_layer(decode_1))
        decode_3 = self.up1(decode_2, res3)
        decode_4 = self.invert2(self.conv_layer(decode_3))
        decode_5 = self.up2(decode_4, res2)
        decode_6 = self.invert3(self.conv_layer(decode_5))
        decode_7 = self.up3(decode_6, res1)
        decode_8 = self.invert4(self.conv_layer(decode_7))
        decode_8 = self.scope(decode_8)
        x = self.outc(decode_8)
        return x


class DySample(nn.Module):
    def __init__(self, in_channels, scale=4, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid(h, h, indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1,
                                                                                                                  1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = feat_e + x

        return out

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 2
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))
        self.conv = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv1 = nn.Conv2d(320, nclass, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, x):
        batchsize, C, width, height = x.size()

        x1 = self.query_conv(x).view(batchsize, -1, width*height).permute(0, 2, 1)
        x2 = self.key_conv(x).view(batchsize, -1, width*height)
        x3 = torch.bmm(x1, x2).view(batchsize, -1, width, height)
        x3 = self.conv(self.pam(x3))

        x4 = self.value_conv(x)
        x5 = self.cam(x4)

        x6 = torch.bmm(x3.view(batchsize, -1, width*height), x5.view(batchsize, -1, width*height).permute(0, 2, 1)).view(batchsize, -1, width, height)

        x7 = self.conv1(torch.cat((x6, self.conv2(x)), dim=1))
        return x7


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW'):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils import LayerNorm, GRN


class ConvV2_concat_Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.conv = nn.Conv2d(dim, 64, 1)
        self.dwconv = nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64)  # depthwise conv
        self.norm = LayerNorm(64, eps=1e-6)
        self.pwconv1 = nn.Linear(64, 256)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(256)
        self.pwconv2 = nn.Linear(256, 64)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class CSE_Block(nn.Module):
    def __init__(self, in_channel, r, w, h):  # r是压缩比
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            # 在这里相当于把宽为80，高为80，通道为128的特征图直接池化成C维的向量（全局池化，并不是只池化一部分，所以要（w,h））
            nn.AvgPool2d((w, h)),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s*x

class Mlp(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.GELU()
        )
        self.proj = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch)
        self.proj_act = nn.GELU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=True),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x