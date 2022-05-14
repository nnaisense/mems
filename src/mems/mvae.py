from typing import Tuple

import torch
from omegaconf import DictConfig, read_write
from torch import nn

from mems.archs import get_arch


class MVAE(torch.nn.Module):
    """A Multimeasurement VAE with a known noise model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        assert len(cfg.sigma) == 1, "len(sigma) > 1 is not supported yet!"
        assert len(cfg.M) == len(cfg.sigma), "len(M) must be equal to len(sigma)!"
        self.M = cfg.M[0]  # Assume there is a single noise scale with M measurements
        self.sigma = cfg.sigma[0]
        self.two_encoders = cfg.get("two_encoders", False)
        if self.two_encoders is True:
            self.encoder = nn.ModuleList([get_arch(cfg.encoder) for _ in ["mu", "logvar"]])
        else:
            self.encoder = nn.ModuleList([get_arch(cfg.encoder)])
        self.decoder = nn.ModuleList([get_arch(cfg.decoder) for _ in self.cfg.M])
        if self.cfg.meta:
            self.metaencoder = get_arch(cfg.metaencoder)
            self.metaout = nn.Linear(cfg.metaencoder.config.out_channels, 1, bias=False)
        else:
            self.metaencoder, self.metaout = None, None

        self.logp, self.kld, self.nu, self.meta, self.mu, self.std = None, None, None, None, None, None
        self.deterministic = True if cfg.beta == 0.0 else False

        input_scale = torch.sqrt(torch.tensor(0.225 ** 2 + self.cfg.sigma[0] ** 2, device=self.device))
        self.register_buffer("input_scale", input_scale)

        self.zdim = cfg.zdim

    @staticmethod
    def finalize_config(cfg):
        with read_write(cfg):
            cfg.encoder.config.in_channels = cfg.M[0] * cfg.input_shape[0]
            cfg.decoder.config.in_channels = cfg.zdim
            cfg.encoder.config.out_channels = cfg.zdim if cfg.two_encoders is True else 2 * cfg.zdim
            cfg.decoder.config.out_channels = cfg.M[0] * cfg.input_shape[0]
            if cfg.meta:
                cfg.metaencoder.config.in_channels = 2 * cfg.M[0] * cfg.input_shape[0]
        return cfg

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        self.mu, self.std = mu, std
        return mu + eps * std

    def encode(self, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        y = y / self.input_scale
        if self.two_encoders is True:
            mu = self.encoder[0](y)
            logvar = self.encoder[1](y)
        else:
            features = self.encoder[0](y)
            mu, logvar = torch.chunk(features, 2, dim=1)

        return mu, logvar

    def forward(self, y: torch.Tensor):
        mu, logvar = self.encode(y)
        if self.deterministic:
            z = mu
            self.mu = mu
        else:
            z = self.reparameterize(mu, logvar)

        if self.cfg.noise_model == "GAUSS":
            nu = self.decoder[0](z)
            self.nu = nu
            sq_error = (y - nu).pow(2)
            logp = (-sq_error / (2 * self.sigma ** 2)).sum(dim=(1, 2, 3))  # ignoring additive constant
            if self.metaencoder is not None:
                meta_input = torch.cat([y / self.input_scale, nu], dim=1)
                meta = -self.metaout(torch.flatten(self.metaencoder(meta_input), start_dim=1))
                meta = meta.squeeze(-1).squeeze(-1) / (2 * self.sigma ** 2)
                self.meta = meta
                logp = logp + meta
        else:
            raise NotImplementedError(self.cfg.noise_model)

        kld = 0 if self.deterministic else -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
        return z, nu, logp, kld

    def elbo(self, y: torch.Tensor):
        _, _, logp, kld = self(y)
        self.logp, self.kld = logp, kld
        return logp - self.cfg.beta * kld

    def make_xy_from_x0(self, x_0: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_0 = x_0.permute(0, 3, 1, 2).contiguous().float() / 255.0 if self.cfg.dataset == "ffhq256" else x_0
        x = torch.cat(self.M * (x_0,), dim=1)  # expand channels from C to M * C with copies
        x = torch.cat(n * (x,), dim=0)
        y = x + self.sigma * torch.randn_like(x)
        return x, y
