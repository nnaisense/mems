# Parts of this code are modified from VDVAE https://github.com/openai/vdvae. See ./vdvae_license.md for the license.

import math
import warnings
from typing import Optional

import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.nn import functional as F


def get_out_size(module_list: nn.ModuleList, in_shape: tuple):
    """Compute the size of the flattened output of a ModuleList."""
    temp_out = torch.zeros((1,) + in_shape)
    for m in module_list:
        temp_out = m(temp_out)
    return temp_out.numel()


def identity(x):
    return x


def get_activation_fn(activation: Optional[str]):

    if activation is None:
        return identity
    if activation.lower() == "silu":
        return F.silu
    elif activation.lower() == "relu":
        return F.relu
    else:
        raise NotImplementedError(activation)


class Resnet(nn.Module):
    def __init__(self, arch_config: DictConfig):
        super().__init__()
        cfg = self.cfg = arch_config
        if "act" not in cfg.keys():
            warnings.warn("No act specified for Resnet. Setting to 'silu'.")
            with open_dict(cfg):
                cfg.act = "silu"
        if "init" not in cfg.keys():
            warnings.warn("No init specified for Resnet. Setting to 'orthogonal'.")
            with open_dict(cfg):
                cfg.init = "orthogonal"
        if "gain" not in cfg.keys():
            warnings.warn("No act specified for Resnet. Setting to '1.414'.")
            with open_dict(cfg):
                cfg.gain = 1.414

        block_specs = parse_resnet_specs_list(cfg.blocks, reverse=cfg.reverse_blocks)
        first_block_width = block_specs[0][-1]
        self.in_conv = get_1x1(cfg.in_channels, first_block_width)
        self.out_conv = get_1x1(block_specs[-1][-1], cfg.out_channels)
        self.out_act = get_activation_fn(cfg.out_act)
        self.act = get_activation_fn(cfg.act)
        blocks = []
        prev_width = first_block_width
        for i, (res, down_rate, block_width) in enumerate(block_specs):
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            if down_rate is None:
                block_type = cfg.get("block_type", "deepbottleneck")
                if block_type == "deepbottleneck":
                    Block = ConvBlock
                elif block_type == "simple":
                    Block = SimpleConvBlock
                else:
                    raise NotImplementedError(cfg.block_type)
                blocks.append(
                    Block(
                        prev_width,
                        block_width,
                        int(block_width * cfg.bottleneck_multiple),
                        block_width,
                        residual=True,
                        act=cfg.act,
                        use_3x3=use_3x3,
                    )
                )
            else:
                blocks.append(ResolutionBlock(down_rate, cfg.reverse_blocks))
            prev_width = block_width if block_width is not None else prev_width

        def _init(m):
            if type(m) is nn.Conv2d:
                if cfg.init.lower() == "orthogonal":
                    torch.nn.init.orthogonal_(m.weight.data, gain=cfg.gain)
                    torch.nn.init.zeros_(m.bias.data)
                elif cfg.init.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=cfg.gain)
                    torch.nn.init.zeros_(m.bias.data)
                elif cfg.init.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=cfg.gain)
                    torch.nn.init.zeros_(m.bias.data)
                elif cfg.init.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                    torch.nn.init.zeros_(m.bias.data)
                elif cfg.init.lower() == "default":
                    pass
                else:
                    raise NotImplementedError(cfg.init)

        # Initialization
        self.apply(_init)
        scale_init = cfg.get("scale_init", False)
        zero_last = cfg.get("zero_last", False)
        L = len(blocks) // 2
        for b in blocks:
            b.custom_init(L, scale_init, zero_last)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_act(self.out_conv(self.act(x)))
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        prev_width: int,
        in_width: int,
        middle_width: int,
        out_width: int,
        residual: bool = False,
        act: str = "silu",
        use_3x3: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.act = get_activation_fn(act)
        if prev_width < in_width:
            self.pad_channels = in_width
        elif prev_width > in_width:
            self.proj_conv = get_1x1(prev_width, in_width)
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width)

    def custom_init(self, L, scale_init, zero_last):
        m = 4
        if scale_init:
            self.c1.weight.data *= math.pow(L, -1 / (2 * m - 2))
            self.c2.weight.data *= math.pow(L, -1 / (2 * m - 2))
            self.c3.weight.data *= math.pow(L, -1 / (2 * m - 2))
        if zero_last:
            self.c4.weight.data *= 0.0

    def forward(self, x):
        if hasattr(self, "pad_channels"):
            x = pad_channels(x, self.pad_channels)
        elif hasattr(self, "proj_conv"):
            x = self.proj_conv(x)
        x_prime = self.c1(self.act(x))
        x_prime = self.c2(self.act(x_prime))
        x_prime = self.c3(self.act(x_prime))
        x_prime = self.c4(self.act(x_prime))
        out = x + x_prime if self.residual else x_prime
        return out


class SimpleConvBlock(nn.Module):
    """Use 2 3x3 Conv layers only. Does not use middle_width, bottleneck_multiple, out_width"""

    def __init__(
        self,
        prev_width: int,
        in_width: int,
        middle_width: int,
        out_width: int,
        residual: bool = False,
        act: str = "silu",
        use_3x3: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.act = get_activation_fn(act)
        if prev_width < in_width:
            self.pad_channels = in_width
        elif prev_width > in_width:
            self.proj_conv = get_1x1(prev_width, in_width)
        self.c1 = get_3x3(in_width, in_width) if use_3x3 else get_1x1(in_width, in_width)
        self.c2 = get_3x3(in_width, in_width) if use_3x3 else get_1x1(in_width, in_width)

    def custom_init(self, L, scale_init, zero_last):
        m = 2
        if scale_init:
            self.c1.weight.data *= math.pow(L, -1 / (2 * m - 2))
        if zero_last:
            self.c2.weight.data *= 0.0

    def forward(self, x):
        if hasattr(self, "pad_channels"):
            x = pad_channels(x, self.pad_channels)
        elif hasattr(self, "proj_conv"):
            x = self.proj_conv(x)
        x_prime = self.c1(self.act(x))
        x_prime = self.c2(self.act(x_prime))
        out = x + x_prime if self.residual else x_prime
        return out


class ResolutionBlock(nn.Module):
    def __init__(self, rate: int, upsample: bool):
        super().__init__()
        self.rate = rate
        self.upsample = upsample

    def custom_init(self, *args, **kwargs):
        pass

    def forward(self, x):
        if self.upsample:
            out = F.interpolate(x, scale_factor=self.rate)
        else:
            out = F.avg_pool2d(x, kernel_size=int(self.rate), stride=int(self.rate), ceil_mode=True)
        return out


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def parse_resnet_specs_list(s: str, reverse: bool) -> list:
    """Returns a list of tuples of [resolution, down_rate, block_width]"""
    layers = []
    segments = s.split(",")[::-1] if reverse else s.split(",")
    for ss in segments:
        if "x" in ss:
            ss, width = ss.split(":")
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None, int(width)) for _ in range(count)]
        elif "d" in ss:
            res, down_rate = [float(a) for a in ss.split("d")]
            layers.append((res, down_rate, None))
        else:
            raise ValueError(ss)
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty
