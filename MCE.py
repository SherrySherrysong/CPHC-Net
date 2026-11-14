import torch
import torch.nn as nn

from .utils import get_activation


__all__ = ["MCE"]


class ConvNormLayer_fuse(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
        d=1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * d if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias,
            dilation=d,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.g = g
        self.padding = padding
        self.bias = bias
        self.d = d

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
                dilation=self.d,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        return self._fuse_bn_tensor()

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class MCE(nn.Module):
    """
    一个轻量级的上下文感受野模块，用于在进入主融合模块前增强特征。
    它使用并行的、不同扩张率的卷积来丰富特征的上下文信息。
    """

    def __init__(self, c1, c2, act="silu"):
        super().__init__()
        c_ = max(c2 // 8, 8)

        self.conv_in = ConvNormLayer_fuse(c1, c2, 1, 1, act=act)

        self.branch1 = ConvNormLayer_fuse(c2, c_, 1, 1, act=act)
        self.branch2 = ConvNormLayer_fuse(c2, c_ * 2, 3, 1, padding=1, d=1, act=act)
        self.branch3 = ConvNormLayer_fuse(c2, c_, 3, 1, padding=2, d=2, act=act)
        self.branch4 = ConvNormLayer_fuse(c2, c_, 3, 1, padding=3, d=3, act=act)
        self.conv_fuse = ConvNormLayer_fuse(c_ * 5, c2, 1, 1, act=act)

    def forward(self, x):
        x = self.conv_in(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return self.conv_fuse(torch.cat([b1, b2, b3, b4], dim=1))
