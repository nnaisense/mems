from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from PIL.Image import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def gen_samples(model, cfg: DictConfig, grid: bool = True) -> Tuple[Optional[List[torch.Tensor]], Optional[Image]]:
    if cfg.mode is None or cfg.nrow < 1 or cfg.ncol < 1:
        return None, None

    model.eval()
    nrow, ncol = cfg.nrow, cfg.ncol
    x_0 = torch.rand((1,) + tuple(model.input_shape))
    x = torch.cat(model.M * (x_0,), dim=1).to(model.device)
    y = x + model.sigma * torch.randn_like(x, device=model.device)
    v = torch.zeros_like(y)
    n = nrow * ncol

    samples = []
    for i in range(n):
        if cfg.mode == "overdamped":
            options = {
                "delta": cfg.eps_scale * model.sigma,
                "steps": cfg.steps,
            }
            y = model.overdamped(y, options)

        elif cfg.mode == "underdamped":
            options = {
                "delta": cfg.eps_scale * model.sigma,
                "friction": cfg.friction,
                "lipschitz": cfg.lipschitz,
                "steps": cfg.steps,
            }
            y, v = model.underdamped(y, v, options)

        elif cfg.mode == "chengetal":
            options = {
                "delta": cfg.eps_scale * model.sigma,
                "friction": cfg.friction,
                "lipschitz": cfg.lipschitz,
                "steps": cfg.steps,
            }
            y, v = model.chengetal(y, v, options)

        else:
            raise NotImplementedError(cfg.model)

        y.detach_().requires_grad_()
        xhat = model.xhat(y)
        dataset = model.vae.cfg.dataset if hasattr(model, "vae") else model.cfg.dataset
        if dataset == "mnist":
            xhat = -xhat + 1
        xhat = xhat.reshape(x_0.shape[0], model.M, x_0.shape[1], x_0.shape[2], x_0.shape[3]).mean(dim=1)
        samples.append(xhat.detach().squeeze(0))
        y.detach_()

    image_grid = to_pil_image(make_grid(samples, nrow=nrow, padding=0, normalize=True, range=(0, 1))) if grid else None
    model.train()
    model.zero_grad()

    return samples, image_grid


def viz_prior(model, cfg: DictConfig, grid: bool = True) -> Tuple[Optional[List[torch.Tensor]], Optional[Image]]:
    model.eval()
    nrow, ncol = cfg.nrow, cfg.ncol
    n = nrow * ncol
    zdim = model.cfg.zdim
    samples = []
    if n == 0:
        return None, None

    for i in range(n):
        z = torch.randn(1, zdim, 1, 1).to(model.device)
        with torch.no_grad():
            nu = model.decoder[0](z)
            nu = nu - nu.min()
            nu = nu / nu.max()
            nu = nu.reshape(
                nu.shape[0], model.M, model.cfg.input_shape[0], model.cfg.input_shape[1], model.cfg.input_shape[2]
            ).mean(dim=1)
            if model.cfg.dataset == "mnist":
                nu = -nu + 1
        samples.append(nu.detach().squeeze(0))

    image_grid = to_pil_image(make_grid(samples, nrow=nrow, padding=0, normalize=True, range=(0, 1))) if grid else None
    model.train()
    model.zero_grad()

    return samples, image_grid
