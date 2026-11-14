import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = ["PDAF", "GateNCS"]


def get_activation(name):
    activations = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


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
            x = self.conv_bn_fused(x)
        else:
            x = self.norm(self.conv(x))
        return self.act(x)

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
        delattr(self, "conv")
        delattr(self, "norm")

    def get_equivalent_kernel_bias(self):
        return self._fuse_bn_tensor()

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        std = (running_var + self.norm.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
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
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            x = self.conv(x)
        else:
            x = self.conv1(x) + self.conv2(x)
        return self.act(x)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        delattr(self, "conv1")
        delattr(self, "conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]), bias3x3 + bias1x1

    @staticmethod
    def _fuse_bn_tensor(branch):
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        std = (running_var + branch.norm.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.conv2 = ConvNormLayer_fuse(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *(bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks))
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(
                hidden_channels, out_channels, 1, 1, bias=bias, act=act
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class CPE(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x):
        return x + self.proj(x)


class AdaptiveMultiScalePool(nn.Module):
    def __init__(self, dim, fc_ratio=4, dilations=(2, 3, 5), pool_size=(20, 20)):
        super().__init__()
        self.num_branches = len(dilations)
        hidden_dim = dim // fc_ratio
        self.conv_in = Conv(dim, hidden_dim, 1)
        self.branches = nn.ModuleList(
            Conv(hidden_dim, hidden_dim, 3, d=d, g=hidden_dim) for d in dilations
        )
        self.weights = nn.Parameter(torch.ones(self.num_branches))
        self.conv_out = Conv(hidden_dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, x):
        residual = x
        reduced = self.conv_in(x)
        features = torch.stack([branch(reduced) for branch in self.branches], dim=1)
        weights = torch.softmax(self.weights, dim=0).view(1, self.num_branches, 1, 1, 1)
        attention = self.conv_out(torch.sum(features * weights, dim=1))
        return self.pool(attention * residual)


class GatedConvFeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(
            hidden_dim // 2,
            hidden_dim // 2,
            3,
            1,
            1,
            groups=hidden_dim // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(hidden_dim // 2)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        x, gate = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.bn(self.dwconv(x))) * gate
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x) + shortcut


class ScaleAwareContextAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        pool_size=(20, 20),
        atten_drop=0.0,
        proj_drop=0.0,
        fc_ratio=4,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.feature_map_sizes = [(160, 160), (80, 80), (40, 40), (20, 20)]
        self.position_encoders = nn.ModuleList(
            CPE(dim) for _ in self.feature_map_sizes
        )
        self.attention_drop = nn.Dropout(atten_drop)
        self.projection_drop = nn.Dropout(proj_drop)
        self.context_pool = AdaptiveMultiScalePool(dim, fc_ratio, pool_size=pool_size)
        self.kv = Conv(dim, 2 * dim, 1)
        self.projection = Conv(dim, dim, 1)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, 1),
            nn.SiLU(),
            nn.Conv2d(dim // fc_ratio, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, scale_index):
        _, _, height, width = x.shape
        if scale_index >= len(self.position_encoders):
            raise IndexError(
                f"scale_index {scale_index} out of range for "
                f"{len(self.position_encoders)} configured scales"
            )

        expected_size = self.feature_map_sizes[scale_index]
        if (height, width) != expected_size:
            print(
                f"Warning: input size {(height, width)} does not match "
                f"scale_index={scale_index} size {expected_size}."
            )

        query_features = self.position_encoders[scale_index](x)
        pooled_features = self.context_pool(x)
        query = rearrange(
            query_features,
            "b (h d) hh ww -> b h (hh ww) d",
            h=self.num_heads,
        )
        key, value = rearrange(
            self.kv(pooled_features),
            "b (kv h d) hh ww -> kv b h (hh ww) d",
            kv=2,
            h=self.num_heads,
        )
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_drop(attention.softmax(dim=-1))
        spatial = rearrange(
            attention @ value,
            "b h (hh ww) d -> b (h d) hh ww",
            hh=height,
            ww=width,
        )
        spatial = self.projection_drop(self.projection(spatial))
        channel = self.channel_attention(self.channel_pool(x)) * x
        return spatial + channel


class ScaleAwareContextBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate)
        self.attention = ScaleAwareContextAttention(dim)
        self.feed_forward = GatedConvFeedForward(dim)

    def forward(self, x, scale_index):
        x = x + self.drop_path(self.attention(x, scale_index))
        return x + self.drop_path(self.feed_forward(x))


class C2fScaleAwareContext(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, scale_index=None, shortcut=True, dropout=0.0):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.blocks = nn.ModuleList(ScaleAwareContextBlock(self.c) for _ in range(n))
        self.scale_index = scale_index
        self.shortcut = shortcut

    def forward(self, x):
        features = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            output = block(features[-1], self.scale_index)
            features.append(output + features[-1] if self.shortcut else output)
        return self.cv2(torch.cat(features, dim=1))


class PDAF(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        block_num=1,
        scale_index=None,
        n=1,
        shortcut=True,
        e=0.5,
        dropout=0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                C2fScaleAwareContext(
                    c1,
                    c2,
                    n=n,
                    e=e,
                    scale_index=scale_index,
                    shortcut=shortcut,
                    dropout=dropout,
                )
            ]
        )
        self.blocks.extend(
            C2fScaleAwareContext(
                c2,
                c2,
                n=n,
                e=e,
                scale_index=scale_index,
                shortcut=shortcut,
                dropout=dropout,
            )
            for _ in range(block_num - 1)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        inter_channels = max(in_channels // reduction_ratio, 16)
        self.context_modeling = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Softmax(dim=2),
        )
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.LayerNorm([inter_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        context_mask = self.context_modeling(x).view(batch, 1, height * width)
        features = x.view(batch, channels, height * width)
        context = torch.bmm(features, context_mask.transpose(1, 2)).unsqueeze(-1)
        return x + self.transform(context)


class GateNCS(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        fused_channels = c3 + 2 * c4
        self.context = GlobalContextBlock(fused_channels)
        self.cv4 = ConvNormLayer_fuse(fused_channels, c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        features = list(self.cv1(x).split((self.c, self.c), dim=1))
        features.extend(block(features[-1]) for block in (self.cv2, self.cv3))
        return self.cv4(self.context(torch.cat(features, dim=1)))
