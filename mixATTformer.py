
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

from thop import profile
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
# from timm.models.layers.helpers import to_2tuple

from torch.nn.init import _calculate_fan_in_and_fan_out
import warnings



class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        #x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        #x = spatial_out * x
        x = channel_out*spatial_out

        return x



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        #out = x * self.ca(x)
        #result = out * self.sa(out)
        result = self.ca(x)*self.sa(x) #x是特征图，初始化此函数的通道数是特征图的通道数。u是原始图像，result*u+u
        return result


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m
def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]
def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'iformer_small': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small.pth'),
    'iformer_base': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base.pth'),
    'iformer_large': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large.pth'),
    'iformer_small_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small_384.pth',
                              input_size=(3, 384, 384), crop_pct=1.0),
    'iformer_base_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base_384.pth',
                             input_size=(3, 384, 384), crop_pct=1.0),
    'iformer_large_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large_384.pth',
                              input_size=(3, 384, 384), crop_pct=1.0),
}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """type: (Tensor, float, float, float, float) -> Tensor
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
       # >>> w = torch.empty(3, 5)
        #>>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class SRMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        # X1 =
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)

class CombinedConv2D(nn.Module):
    def __init__(self, in_channels=3):
        super(CombinedConv2D, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=3, stride=1, kernel_size=5, padding=2)#out_channels=10
        self.bayarConv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=3, stride=1, kernel_size=5, padding=2)
        self.SRMConv2d = SRMConv2D(
            in_channels=3, out_channels=3, stride=1, padding=2)
        self.att = CBAM(in_planes=9,reduction=3)

        self.convcon = nn.Conv2d(in_channels= 12,out_channels=6,kernel_size=5,padding=2)
        self.spatialatt = SpatialAttention(kernel_size=7)
    def forward(self, X):
        X1 = F.relu(self.conv2d(X))
        X2 = F.relu(self.bayarConv2d(X))
        X3 = F.relu(self.SRMConv2d(X))
        X4 = torch.cat([X1, X2, X3], dim=1)
        #X4 = X1+X2+X3#（3，224，224）。X1之后加一个注意力模块，再和X4级联。
        Xnoise = X2+X3
        Xnoise = Xnoise*self.spatialatt(Xnoise)#（3，224，224）

        X_att = self.att(X4)#（12，224，224）
        X_att = self.convcon(X_att)
        X_att = X_att*X1+X1#（6，224，224）
        return X_att


class CombinedConv2D2(nn.Module):
    def __init__(self, in_channels=3):
        super(CombinedConv2D2, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=10, stride=1, kernel_size=5, padding=2)  # out_channels=10
        self.bayarConv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=3, stride=1, kernel_size=5, padding=2)
        self.SRMConv2d = SRMConv2D(
            in_channels=3, out_channels=3, stride=1, padding=2)
        self.spatialatt = SpatialAttention(kernel_size=7)

    def forward(self, X):
        X1 = F.relu(self.conv2d(X))
        X2 = F.relu(self.bayarConv2d(X))
        X3 = F.relu(self.SRMConv2d(X))
        Xnoise = self.spatialatt(X2 + X3)# （1，224，224）
        X4 = self.conv2d(Xnoise)
        X_att = X1 * X4 + X1 # （10，224，224）
        return X_att

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=16, stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        # kernel_size = to_2tuple(kernel_size)
        # stride = to_2tuple(stride)
        # padding = to_2tuple(padding)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x
'''
def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

class IMTFE(nn.Module):
    # **********  *********
    def __init__(self, in_channel=3,):
        super(IMTFE, self).__init__()

        self.relu = nn.ReLU()
        self.device = device

        ## Initialisation 论文里面是10+3+3，这里是4+3+9？

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).to(self.device)
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).to(self.device)
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data = torch.load('IMTFEv4.pt')['SRMConv2D.weight']

        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, H, W = x.shape

        # Normalization
        x = x / 255. * 2 - 1

        ## **Bayar constraints**
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        return last_block
'''
class FirstPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, kernel_size=3, stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.gelu1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        return xa


class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  pool_size=pool_size, )

        self.conv_fuse = nn.Conv2d(low_dim + high_dim * 2, low_dim + high_dim * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim * 2)
        self.proj = nn.Conv2d(low_dim + high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                 attn=Mixer,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         attention_head=attention_head, pool_size=pool_size, )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixInTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=None, depths=None,
                 num_heads=None, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 attention_heads=None,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 checkpoint_path=None,
                 **kwargs,
                 ):

        super().__init__()
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)

        self.num_classes = num_classes
        # ==============
        #self.IMTFE = IMTFE(in_channel=in_chans)
        self.combinedConv = CombinedConv2D(in_channels=in_chans)
        # =================
        self.FAD = FAD_Head(size=img_size)
        #self.att = CBAM(in_planes=12)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.patch_embed = FirstPatchEmbed(in_chans=18, embed_dim=embed_dims[0])#gai1101

        self.num_patches1 = num_patches = img_size // 4  # 为什么除以4，两次卷积后大小变为1/4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[0]))
        self.blocks1 = nn.Sequential(*[
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, )
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            # )
            for i in range(0, st2_idx)])

        self.patch_embed2 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[1]))
        self.blocks2 = nn.Sequential(*[
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, )
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, channel_layer_scale=channel_layer_scale,
            # )
            for i in range(st2_idx, st3_idx)])

        self.patch_embed3 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[2]))
        self.blocks3 = nn.Sequential(*[
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(st3_idx, st4_idx)])

        self.patch_embed4 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[3]))
        self.blocks4 = nn.Sequential(*[
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(st4_idx, depth)])

        self.norm = norm_layer(embed_dims[-1])
        # Classifier head(s)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # set post block, for example, class attention layers

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)

        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def forward_features(self, x):
        x1 = self.combinedConv(x)
        #print(x1.shape)#(16,16,224,224) (16,3,224,224)
        x2 = self.FAD(x)
        #print(x2.shape)#(N,12,224,224)
        x = torch.cat((x1,x2),dim=1) #修改FirstPatchEmbed(in_chans=16
        #print(x.shape)#(16,15,224,224)
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        # print(x.shape)

        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W)
        x = self.blocks1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W)
        x = self.blocks2(x)

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W)
        x = self.blocks3(x)

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W)
        x = self.blocks4(x)
        x = x.flatten(1, 2)

        x = self.norm(x)
        return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


num_classes = 2

def mixATT_iformer_small(pretrained=False, **kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3, 3, 9, 3]
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3

    model = MixInTransformer(img_size=224, num_classes=num_classes,
                                 depths=depths,
                                 embed_dims=embed_dims,
                                 num_heads=num_heads,
                                 attention_heads=attention_heads,
                                 use_layer_scale=True, layer_scale_init_value=1e-6,
                                 **kwargs)
    model.default_cfg = default_cfgs['iformer_small']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def gather_nd(self, params, indices):
    ''' 4D example params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices returns: tensor shaped [m_1, m_2, m_3, m_4] ND_example params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices returns: tensor shaped [m_1, ..., m_1] '''
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        # idx = np.stack([np.arange(log_probs.shape[0]), target], axis=1)
        # nll_loss = torch.gather_nd(params=-log_probs, indices=idx)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = np.to_tensor(1 - drop_prob)
    shape = (np.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + np.rand(shape, dtype=x.dtype)
    random_tensor = np.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output



if __name__ == '__main__':

    #model = mixATT_iformer_small()
    #model.to('cuda')
    #print(model)
    x = plt.imread('../analysis/1_0.png')
    #summary(model, input_size=(3,224,224), batch_size=1, device='cuda')
    '''
#FLOPs: 4.71 GFLOPs
#Params: 18.95 M
    input = torch.randn(1, 3, 224, 224)
    model2 = mixATT_iformer_small()
    flops, params = profile(model2, inputs=(input,),verbose=False)
    print(f'FLOPs: {flops / 1e9:.2f} GFLOPs')  # 将FLOPs转换为GigaFLOPs （即乘以1e-9）
    print(f'Params: {params / 1e6:.2f} M')  # 将参数量转换为百万
    '''
    input3 = torch.randn(2, 16, 224, 224)
    model3 = CBAM(16)
    y = model3(input3)
    print(y.shape)



