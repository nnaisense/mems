# U2net from https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py
# We switch to silu from relu, remove batch norm & remove some stages. We also only use the "fused" map output.
import math

import torch
import torch.nn as nn

__all__ = ["u2net_cifar_wide", "u2net_ffhq256"]


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class CONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1, bias=True):
        super(CONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, bias=bias)
        self.silu_s1 = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu_s1(self.conv_s1(x))


class RSU(nn.Module):
    def __init__(self, name, bias, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.bias = bias
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.convin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f"conv{height}")(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, "downsample")(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f"conv{height}d")(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f"conv{height}")(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module("convin", CONV(in_ch, out_ch, bias=self.bias))
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module("conv1", CONV(out_ch, mid_ch, bias=self.bias))
        self.add_module("conv1d", CONV(mid_ch * 2, out_ch, bias=self.bias))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f"conv{i}", CONV(mid_ch, mid_ch, dilate=dilate, bias=self.bias))
            self.add_module(f"conv{i}d", CONV(mid_ch * 2, mid_ch, dilate=dilate, bias=self.bias))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f"conv{height}", CONV(mid_ch, mid_ch, dilate=dilate, bias=self.bias))


class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch, bias=True):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self.bias = bias
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 5:
                x1 = getattr(self, f"stage{height}")(x)
                x2 = unet(getattr(self, "downsample")(x1), height + 1)
                x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f"stage{height}")(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f"side{h}")(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, "outconv")(x)
            maps.insert(0, x)
            return maps  # [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps[0]  # original: return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], self.bias, *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, 3, padding=1, bias=self.bias))
        # build fuse layer
        self.add_module("outconv", nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1, bias=self.bias))


def u2net_cifar_wide(cfg):
    in_ch, out_ch, f, bias = cfg.in_channels, cfg.out_channels, cfg.factor, cfg.get("bias", True)
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (6, in_ch, 64 * f, 64 * f), -1],
        "stage2": ["En_2", (5, 64 * f, 128 * f, 128 * f), -1],
        "stage3": ["En_3", (4, 128 * f, 128 * f, 256 * f), -1],
        "stage4": ["En_4", (3, 256 * f, 256 * f, 512 * f), -1],
        "stage5": ["En_5", (3, 512 * f, 256 * f, 512 * f), 512 * f],
        "stage4d": ["De_4", (3, 1024 * f, 128 * f, 256 * f), 256 * f],
        "stage3d": ["De_3", (3, 512 * f, 128 * f, 128 * f), 128 * f],
        "stage2d": ["De_2", (5, 256 * f, 128 * f, 64 * f), 64 * f],
        "stage1d": ["De_1", (6, 128 * f, 64 * f, 64 * f), 64 * f],
    }
    return U2NET(cfgs=full, out_ch=out_ch, bias=bias)


def u2net_ffhq256(cfg):
    in_ch, out_ch, f, bias = cfg.in_channels, cfg.out_channels, cfg.factor, cfg.get("bias", True)
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (6, in_ch, 64 * f, 64 * f), -1],
        "stage2": ["En_2", (5, 64 * f, 128 * f, 128 * f), -1],
        "stage3": ["En_3", (4, 128 * f, 128 * f, 256 * f), -1],
        "stage4": ["En_4", (3, 256 * f, 256 * f, 512 * f), -1],
        "stage5": ["En_5", (3, 512 * f, 256 * f, 512 * f), 512 * f],
        "stage4d": ["De_4", (3, 1024 * f, 128 * f, 256 * f), 256 * f],
        "stage3d": ["De_3", (3, 512 * f, 128 * f, 128 * f), 128 * f],
        "stage2d": ["De_2", (5, 256 * f, 128 * f, 64 * f), 64 * f],
        "stage1d": ["De_1", (6, 128 * f, 64 * f, 64 * f), 64 * f],
    }
    return U2NET(cfgs=full, out_ch=out_ch, bias=bias)
