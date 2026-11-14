import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HCAHyperedgeGen",
    "HCAHGConv",
    "HCAComputation",
    "C3HCA",
    "HCA",
]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def fuse_conv_and_bn(conv, bn):
    """Fuse a Conv2d layer and its following BatchNorm2d layer."""
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    fused.requires_grad_(False)

    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias + (bias - bn.running_mean) * scale)
    return fused


class Conv(nn.Module):
    """Standard convolution with normalization and activation."""

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

    def convert_to_deploy(self):
        if hasattr(self, "bn"):
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
            delattr(self, "bn")
            self.forward = self.forward_fuse


class DSConv_YOLO13(nn.Module):
    """Depthwise separable convolution used by the HyperACE blocks."""

    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=c_in,
            bias=bias,
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DSBottleneck(nn.Module):
    """Bottleneck built from depthwise separable convolutions."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv_YOLO13(c1, c_, k1)
        self.cv2 = DSConv_YOLO13(c_, c2, k2, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(nn.Module):
    """C3-style block using DSBottleneck modules."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(
                DSBottleneck(c_, c_, shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class FuseModule(nn.Module):
    """Align and fuse three feature-map scales."""

    def __init__(self, c_in, channel_adjust):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = Conv((4 if channel_adjust else 3) * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        return self.conv_out(torch.cat((x1_ds, x[1], x3_up), dim=1))


class HCAHyperedgeGen(nn.Module):
    """Generate an N x K incidence matrix from contextual and spatial clustering."""

    def __init__(
        self,
        node_dim,
        num_hyperedges,
        num_heads=4,
        dropout=0.1,
        context="both",
        spatial_sigma=0.25,
        record_affinities=False,
    ):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads.")
        if num_hyperedges < 1:
            raise ValueError("num_hyperedges must be positive.")
        if spatial_sigma <= 0:
            raise ValueError("spatial_sigma must be positive.")
        if context not in ("mean", "max", "both"):
            raise ValueError("context must be 'mean', 'max', or 'both'.")

        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context
        self.spatial_sigma = spatial_sigma
        self.record_affinities = record_affinities

        self.prototype_base = nn.Parameter(torch.empty(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)

        context_dim = 2 * node_dim if context == "both" else node_dim
        self.context_net = nn.Linear(context_dim, num_hyperedges * node_dim)
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.spatial_params = nn.Parameter(self._init_spatial_params(num_hyperedges))
        self.dropout = nn.Dropout(dropout)

        self.last_context_logits = None
        self.last_spatial_logits = None
        self.last_incidence = None

    @staticmethod
    def _init_spatial_params(num_hyperedges):
        rows = max(1, int(math.floor(math.sqrt(num_hyperedges))))
        cols = int(math.ceil(num_hyperedges / rows))
        centers = [
            ((col + 0.5) / cols, (row + 0.5) / rows)
            for row in range(rows)
            for col in range(cols)
        ][:num_hyperedges]
        centers = torch.tensor(centers, dtype=torch.float32)
        return torch.log(centers / (1.0 - centers))

    def _context_features(self, x):
        if self.context == "mean":
            return x.mean(dim=1)
        if self.context == "max":
            return x.amax(dim=1)
        return torch.cat((x.mean(dim=1), x.amax(dim=1)), dim=-1)

    def _contextual_logits(self, x):
        batch, num_nodes, channels = x.shape
        offsets = self.context_net(self._context_features(x)).view(
            batch, self.num_hyperedges, channels
        )
        prototypes = self.prototype_base.unsqueeze(0) + offsets
        nodes = self.node_proj(x)

        nodes = nodes.view(batch, num_nodes, self.num_heads, self.head_dim)
        prototypes = prototypes.view(
            batch, self.num_hyperedges, self.num_heads, self.head_dim
        )
        return torch.einsum("bnhd,bkhd->bhnk", nodes, prototypes).mean(dim=1) / math.sqrt(
            self.head_dim
        )

    def _spatial_logits(self, height, width, device, dtype):
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        node_coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        centers = torch.sigmoid(self.spatial_params).to(dtype=dtype)
        squared_distance = (node_coordinates[:, None, :] - centers[None, :, :]).square().sum(-1)
        return -squared_distance / (2.0 * self.spatial_sigma**2)

    def forward(self, x, height, width):
        batch, num_nodes, _ = x.shape
        if num_nodes != height * width:
            raise ValueError(
                f"Expected {height * width} nodes for a {height}x{width} feature map, "
                f"but received {num_nodes}."
            )

        context_logits = self._contextual_logits(x)
        spatial_logits = self._spatial_logits(height, width, x.device, x.dtype)
        spatial_logits = spatial_logits.unsqueeze(0).expand(batch, -1, -1)
        incidence = F.softmax(self.dropout(context_logits + spatial_logits), dim=1)

        if self.record_affinities:
            self.last_context_logits = context_logits.detach()
            self.last_spatial_logits = spatial_logits.detach()
            self.last_incidence = incidence.detach()

        return incidence


class HCAHGConv(nn.Module):
    """Apply vertex-to-edge and edge-to-vertex propagation."""

    def __init__(
        self,
        embed_dim,
        num_hyperedges=8,
        num_heads=4,
        dropout=0.1,
        context="both",
        spatial_sigma=0.25,
        record_affinities=False,
    ):
        super().__init__()
        self.edge_generator = HCAHyperedgeGen(
            embed_dim,
            num_hyperedges,
            num_heads,
            dropout,
            context,
            spatial_sigma,
            record_affinities,
        )
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, x, height, width):
        incidence = self.edge_generator(x, height, width)
        hyperedges = self.edge_proj(torch.bmm(incidence.transpose(1, 2), x))
        updated_nodes = self.node_proj(torch.bmm(incidence, hyperedges))
        return x + updated_nodes


class HCAComputation(nn.Module):
    """Apply HCA hypergraph computation to a BCHW feature map."""

    def __init__(
        self,
        embed_dim,
        num_hyperedges=8,
        num_heads=8,
        dropout=0.1,
        context="both",
        spatial_sigma=0.25,
        record_affinities=False,
    ):
        super().__init__()
        self.hgnn = HCAHGConv(
            embed_dim,
            num_hyperedges,
            num_heads,
            dropout,
            context,
            spatial_sigma,
            record_affinities,
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens, height, width)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class C3HCA(nn.Module):
    """CSP-style feature block containing one HCA computation branch."""

    def __init__(
        self,
        c1,
        c2,
        e=1.0,
        num_hyperedges=8,
        context="both",
        spatial_sigma=0.25,
        record_affinities=False,
    ):
        super().__init__()
        hidden_channels = int(c2 * e)
        if hidden_channels % 16 != 0:
            raise ValueError("The HCA hidden dimension must be a multiple of 16.")

        self.cv1 = Conv(c1, hidden_channels, 1, 1)
        self.cv2 = Conv(c1, hidden_channels, 1, 1)
        self.m = HCAComputation(
            embed_dim=hidden_channels,
            num_hyperedges=num_hyperedges,
            num_heads=hidden_channels // 16,
            dropout=0.1,
            context=context,
            spatial_sigma=spatial_sigma,
            record_affinities=record_affinities,
        )
        self.cv3 = Conv(2 * hidden_channels, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class HCA(nn.Module):
    """Hypergraph Contextual Aggregation for three aligned feature levels."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1.0,
        context="both",
        channel_adjust=False,
        spatial_sigma=0.25,
        record_affinities=False,
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7)
            if dsc3k
            else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        branch_args = dict(
            e=e2,
            num_hyperedges=num_hyperedges,
            context=context,
            spatial_sigma=spatial_sigma,
            record_affinities=record_affinities,
        )
        self.branch1 = C3HCA(self.c, self.c, **branch_args)
        self.branch2 = C3HCA(self.c, self.c, **branch_args)

    def forward(self, x):
        fused = self.fuse(x)
        features = list(self.cv1(fused).chunk(3, dim=1))
        branch1 = self.branch1(features[1])
        branch2 = self.branch2(features[1])
        features.extend(block(features[-1]) for block in self.m)
        features[1] = branch1
        features.append(branch2)
        return self.cv2(torch.cat(features, dim=1))
