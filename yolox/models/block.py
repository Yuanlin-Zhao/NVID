import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=math.gcd(c1, c2), dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class FREConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = FConv(c1, c2, k, s, p=p, g=g, act=False)

    def forward_fuse(self, x):

        return self.act(self.conv(x))

    def forward(self, x):

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + id_out)

    def get_equivalent_kernel_bias(self):

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):

        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):

        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels , groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
        self.conv = nn.Conv2d(40, 20, 1)
        self.conv2 = nn.Conv1d(20, 40, 1)
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class PDA(nn.Module):
    def __init__(self, channels, groups=8, mode=""):
        super(PDA, self).__init__()
        self.groups = groups
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv1x1 = DeformConv(channels // self.groups, channels // self.groups, kernel_size=(1, 1), stride=1,
                                      padding=0, groups=channels // self.groups)
        self.conv3x3 = DeformConv(channels // self.groups, channels // self.groups, kernel_size=(3, 3), stride=1,
                                      padding=1, groups=channels // self.groups)
        self.channels = channels
        self.mode = mode
        self.NVIDxinConv = nn.Conv2d(320, channels // self.groups, kernel_size=1)
        self.NVIDxoutConv = nn.Conv2d(channels // self.groups, 320, kernel_size=1)

    def forward(self, x):
        if self.mode == "NVIDx":
            x = self.NVIDxinConv(x)
            x = self.conv1x1(x)
            x = self.conv3x3(x)
            x = self.NVIDxoutConv(x)
            return x
        else:
            b, c, h, w = x.size()
            group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
            x_h = self.pool_h(group_x)
            x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
            hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
            x_h, x_w = torch.split(hw, [h, w], dim=2)
            x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
            x2 = self.conv3x3(group_x)
            x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
            x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
            x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
            x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
            weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
            return (group_x * weights.sigmoid()).reshape(b, c, h, w)

######################################## CorrelatedFrameChannelAttention ########################################
class CFCA(nn.Module):#CorrelatedFrameChannelAttention
    def __init__(self, in_channels, frame, train_batch, mode):
        super(CFCA, self).__init__()
        # if mode = 'NVIDx':
        self.mode = mode
        if mode == 'NVIDx':
            self.dim = in_channels
            self.frame = frame
            self.train_batch = train_batch
            self.ChannelConv1x1 = nn.Sequential(nn.Conv2d(
                self.dim * 2, self.dim * 4, kernel_size=1,
                stride=1),
                nn.BatchNorm2d(self.dim * 4),
                nn.GELU())

            self.ChannelConv3x3 = nn.Sequential(nn.Conv2d(
                self.dim * 2, self.dim * 4, kernel_size=3,
                stride=1, padding=1),
                nn.BatchNorm2d(self.dim * 4),
                nn.GELU())
            self.CatConv = nn.Conv2d(self.dim * 8, self.dim * 4, kernel_size=1)
            self.BatchNorm2d = nn.BatchNorm2d(self.dim * 4)
            self.act = nn.GELU()
        else:
            self.dim = in_channels
            self.frame = frame
            self.train_batch = train_batch
            self.ChannelConv1x1 = nn.Sequential(nn.Conv2d(
                self.dim, self.dim * 2, kernel_size=1,
                stride=1),
                nn.BatchNorm2d(self.dim * 2),
                nn.GELU())

            self.ChannelConv3x3 = nn.Sequential(nn.Conv2d(
                self.dim, self.dim * 2, kernel_size=3,
                stride=1, padding=1),
                nn.BatchNorm2d(self.dim * 2),
                nn.GELU())
            self.CatConv = nn.Conv2d(self.dim * 4, self.dim * 2, kernel_size=1)
            self.BatchNorm2d = nn.BatchNorm2d(self.dim * 2)
            self.act = nn.GELU()
    def forward(self, x):
        [B, C, H, W]= x.size()
        batch_frame = int(B // self.frame)
        if batch_frame == 0:
            batch_frame = batch_frame + 1
        if self.mode == 'NVIDx':D:\axianyu\1-20\new\fusion_detect\VOCdevkit\VOC2007\LLVIP\Annotations
            x = x.view(batch_frame, B * self.dim *2, H, W)
        else:
            x = x.view(batch_frame, B * self.dim , H, W)
        x1 = self.ChannelConv1x1(x)
        x2 = self.ChannelConv3x3(x)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.CatConv(x3)
        x3 = self.BatchNorm2d(x3)
        x3 = self.act(x3)
        x1, x2 = torch.split(x3, [C, C], dim=1)
        return x1 * x2

#####################################correlated frame temporal attention####################

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x

class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = FConv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class EfficientMixformer(nn.Module):  # EST
    def __init__(self, inchannels, hiddenchannels, outchannels, mode='NVID'):
        super().__init__()
        self.mode = mode
        self.in_channels = inchannels
        self.hidden_channels = hiddenchannels
        self.out_channels = outchannels
        if self.mode == 'NVIDx':
            self.tr1 = TransformerBlock(self.in_channels, self.hidden_channels, 4, 1)
            self.tr2 = TransformerBlock(self.hidden_channels, self.out_channels, 4, 1)
            self.outconv = FConv(self.in_channels, self.out_channels, k=1)
        else:
            self.tr1 = TransformerBlock(self.in_channels, self.out_channels, 4, 1)
            self.outconv = FConv(self.in_channels, self.out_channels, k=1)


    def forward(self, x):
        y = x.clone()
        if self.mode == 'NVIDx':
           x = self.tr1(x)
           x = self.tr2(x)
           y = self.outconv(y)
        else:
           x = self.tr1(x)
           y = self.outconv(y)
        return y + x


class CFTA(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, frame=8, mode=''):
        super(CFTA, self).__init__()
        self.mode = mode
        if mode == 'NVIDx':
            print("NVIDx")
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.frame = frame

            self.in_conv = nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=3,
                                     stride=1, padding=3, dilation=1)

            self.second_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=5,
                                         stride=1, padding=7, dilation=3)

            self.three_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=7,
                                        stride=1, padding=12, dilation=5)
            self.hiddenbn = nn.BatchNorm1d(self.hidden_channels)
            self.act = nn.GELU()

            self.tr = EfficientMixformer(self.hidden_channels * 2, self.hidden_channels * 2, self.out_channels * 2, mode=self.mode)
        else:
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.frame = frame

            self.in_conv = nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=3,
                                     stride=1, padding=3, dilation=1)

            self.second_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=5,
                                         stride=1, padding=7, dilation=3)

            self.three_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=7,
                                        stride=1, padding=12, dilation=5)
            self.hiddenbn = nn.BatchNorm1d(self.hidden_channels)
            self.act = nn.GELU()
            self.tr = EfficientMixformer(self.hidden_channels, self.hidden_channels* 2, self.out_channels, mode=self.mode)
            #self.tr = TransformerBlock(self.hidden_channels, self.out_channels, 4, 1, mode=self.mode)


    def forward(self, x):
        batch, c, h, w = x.size()
        y = x.clone()
        b = int(x.size(0)//self.frame)
        if (b % 2) != 0 or b == 0:
            b = b + 1
        if self.mode == 'NVIDx':
            x = x.view(b, self.in_channels,  -1)
        else:
            x = x.view(b, self.in_channels, -1)
        x = self.in_conv(x)
        x = self.hiddenbn(x)
        x = self.act(x)
        x = self.second_conv(x)
        x = self.hiddenbn(x)
        x = self.act(x)
        x = self.three_conv(x)
        x = self.hiddenbn(x)
        x = self.act(x)
        x = x.view(b, -1)
        if self.mode == 'NVIDx':
            x = x.reshape(batch, self.hidden_channels * 2, h, w)
        else:
            x = x.reshape(batch, self.hidden_channels , h, w)
        x = self.tr(x)

        return y * x


class TFM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, frame=8, mode='NVID'):
        super(TFM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.frame = frame
        self.mode = mode
        self.CFCA = CFCA(in_channels=self.in_channels, frame=self.frame, train_batch=8, mode=self.mode)

        self.CFTA = CFTA(in_channels=self.in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                         frame=self.frame, mode=self.mode)
    def forward(self, x):

        return self.CFTA(self.CFCA(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class FRepBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = FREConv(c1, c_, k=3,  p=1)
        self.cv2 = FREConv(c_, c2, k=3,  p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class FRECSP(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = FConv(c1, c_, 1, 1)
        self.cv2 = FConv(c1, c_, 1, 1)
        self.cv3 = FConv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(FRepBottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class FRELayer(FRECSP):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FRepBottleneck(c_, c_) for _ in range(n)))
