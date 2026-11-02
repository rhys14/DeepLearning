"""Building blocks for GalaxyDETR: backbones, 2D positional encoding, transformer layers, MLP head."""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet-18-style residual block: two 3x3 convs plus a skip connection."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                      nn.BatchNorm2d(c_out))

    def forward(self, x):
        idt = x if self.skip is None else self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + idt)


class SmallBackbone(nn.Module):
    """From-scratch CNN: stem(/2) -> stage1(/4) -> stage2(/8) -> stage3(/16)."""
    def __init__(self, in_ch=3, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, c = [], widths[0]
        for w in widths[1:]:
            layers += [BasicBlock(c, w, stride=2), BasicBlock(w, w)]
            c = w
        self.body = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.body(self.stem(x))


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 from timm, returning only the stride-16 feature map.

    Variants: mobilenetv4_conv_small (default), mobilenetv4_conv_medium, mobilenetv4_hybrid_medium, ...
    The stride-32 stage is never built. in_ch != 3 is handled by timm, which adapts the
    pretrained first conv to the new channel count.
    """
    def __init__(self, name='mobilenetv4_conv_small', pretrained=True, in_ch=3, stride=16):
        super().__init__()
        probe = timm.create_model(name, pretrained=False, features_only=True)
        idx = probe.feature_info.reduction().index(stride)
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True,
                                      out_indices=(idx,), in_chans=in_ch)
        self.out_channels = self.body.feature_info.channels()[-1]

    def forward(self, x):
        return self.body(x)[-1]


def build_backbone(name='mobilenetv4_conv_small', pretrained=True, in_ch=3):
    """'custom' -> SmallBackbone; any timm mobilenetv4_* name -> MobileNetV4Backbone."""
    if name == 'custom':
        return SmallBackbone(in_ch)
    return MobileNetV4Backbone(name, pretrained, in_ch)


def sine_pos_embed(h, w, dim, device, temperature=10000.0):
    """Returns (h*w, dim): half the channels encode x, half encode y."""
    assert dim % 4 == 0, 'dim must be divisible by 4'
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    y = (y + 0.5) / h * 2 * math.pi
    x = (x + 0.5) / w * 2 * math.pi
    d = dim // 4
    omega = 1.0 / temperature ** (torch.arange(d, device=device, dtype=torch.float32) / d)
    y = y.flatten()[:, None] * omega[None]
    x = x.flatten()[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class EncoderLayer(nn.Module):
    """Self-attention over image tokens + FFN."""
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, src, pos):
        s = self.n1(src)
        src = src + self.drop(self.attn(s + pos, s + pos, s)[0])
        return src + self.drop(self.ff(self.n2(src)))


class DecoderLayer(nn.Module):
    """Self-attention among object queries, cross-attention into image tokens, FFN."""
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, tgt, memory, pos, query_pos):
        t = self.n1(tgt)
        tgt = tgt + self.drop(self.self_attn(t + query_pos, t + query_pos, t)[0])
        t = self.n2(tgt)
        tgt = tgt + self.drop(self.cross_attn(t + query_pos, memory + pos, memory)[0])
        return tgt + self.drop(self.ff(self.n3(tgt)))


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers):
        super().__init__()
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x
