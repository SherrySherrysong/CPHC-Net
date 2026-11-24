"""Fusion layers adapted from the HyperACE module in the research code."""

import torch
from torch import nn

from HCA import Conv


class Concat(nn.Module):
    def forward(self, features):
        return torch.cat(features, dim=1)


class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """

    def __init__(self, in_channels, channel_adjust=False):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity()

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out
