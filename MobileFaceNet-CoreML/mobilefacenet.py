"""
Reference MobileFaceNet architecture (Chen et al., 2018 — arXiv:1804.07573).

This is used ONLY to produce a self-contained, runnable reference ONNX so the
conversion + verification pipeline can be proven end-to-end WITHOUT the
production weights (see README, "Why a reference model?"). The production model
is InsightFace `w600k_mbf.onnx`, which is byte-compatible with this pipeline
(same I/O: [1,3,112,112] float -> [1,512] float). convert.py does not depend on
this file at all; it consumes a plain .onnx.

The layer topology below matches the canonical MobileFaceNet: a stride-2 stem,
a depthwise layer, five bottleneck stages (expansion / depthwise / linear
projection with residuals), a 1x1 expansion to 512, a *linear* global depthwise
conv (GDConv, no ReLU) that replaces global average pooling, and a 1x1 linear
projection to the 512-d embedding. PReLU is used throughout, as in the paper.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BN -> (optional) PReLU. groups>1 gives a depthwise conv."""
    def __init__(self, cin, cout, k, s, p, groups=1, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout) if use_act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act is not None else x


class Bottleneck(nn.Module):
    """Inverted residual: 1x1 expand -> 3x3 depthwise -> 1x1 linear project."""
    def __init__(self, cin, cout, stride, expansion):
        super().__init__()
        hidden = cin * expansion
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvBlock(cin, hidden, 1, 1, 0)
        self.dw = ConvBlock(hidden, hidden, 3, stride, 1, groups=hidden)
        self.project = ConvBlock(hidden, cout, 1, 1, 0, use_act=False)  # linear

    def forward(self, x):
        out = self.project(self.dw(self.expand(x)))
        return x + out if self.use_res else out


def _stage(cin, cout, n, stride, expansion):
    layers = [Bottleneck(cin, cout, stride, expansion)]
    for _ in range(n - 1):
        layers.append(Bottleneck(cout, cout, 1, expansion))
    return nn.Sequential(*layers)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = ConvBlock(3, 64, 3, 2, 1)                 # 112 -> 56
        self.dw = ConvBlock(64, 64, 3, 1, 1, groups=64)       # 56
        self.stage1 = _stage(64, 64, n=5, stride=2, expansion=2)   # 56 -> 28
        self.stage2 = _stage(64, 128, n=1, stride=2, expansion=4)  # 28 -> 14
        self.stage3 = _stage(128, 128, n=6, stride=1, expansion=2) # 14
        self.stage4 = _stage(128, 128, n=1, stride=2, expansion=4) # 14 -> 7
        self.stage5 = _stage(128, 128, n=2, stride=1, expansion=2) # 7
        self.conv_exp = ConvBlock(128, 512, 1, 1, 0)          # 7x7x512
        # Linear global depthwise conv (GDConv) — replaces global avg pool.
        self.gdconv = ConvBlock(512, 512, 7, 1, 0, groups=512, use_act=False)
        self.linear = nn.Conv2d(512, embedding_size, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(embedding_size)

    def forward(self, x):
        x = self.dw(self.stem(x))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.stage4(x); x = self.stage5(x)
        x = self.conv_exp(x)
        x = self.gdconv(x)
        x = self.bn(self.linear(x))
        return torch.flatten(x, 1)   # [N, 512]


if __name__ == "__main__":
    m = MobileFaceNet().eval()
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(1, 3, 112, 112))
    print(f"params: {n/1e6:.2f}M   output: {tuple(y.shape)}")
