from omegaconf import DictConfig
from torch import nn

from mems.archs.resnet import Resnet
from mems.archs.u2net import u2net_cifar_wide, u2net_ffhq256

ARCHS = {
    "Resnet": Resnet,
    "U2NetCifarWide": u2net_cifar_wide,
    "U2NetFFHQ256": u2net_ffhq256,
}


def get_arch(cfg: DictConfig) -> nn.Module:
    try:
        ARCH = ARCHS[cfg["class"]]
    except KeyError:
        raise KeyError(f"{cfg['class']} is not one of valid archs: {ARCHS})")
    arch_config = cfg["config"]
    return ARCH(arch_config)
