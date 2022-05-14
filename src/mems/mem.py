import math

import torch
from omegaconf import DictConfig, read_write
from pytorch_lightning import LightningModule
from torch import nn

from .archs import get_arch
from .utils import get_optimizer


class MEM(LightningModule):
    """
    Multimeasurement Energy Model with M measurements.
    Note that the energy (phi) is in Y space (y is a stack of noisy observations), so all methods expect NOISY data.
    The expected input shape is (batch, channels * M, height, width).
    Clean inputs (x_0) are only expected during training, by self.training_step().
    """

    def __init__(self, cfg: DictConfig):
        """
        Create a Multi-scale Energy Model.
        Args:
            cfg: The model configuration
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.M = cfg.M[0]  # Assume there is a single noise level with M measurements
        self.phi = get_arch(cfg.phi)
        self.input_shape = cfg.input_shape
        self.sync_log = True if cfg.training.get("sync_log", None) is True else False
        self.example_input_array = torch.zeros(
            1, self.M * self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        input_scale = torch.sqrt(torch.tensor(0.225 ** 2 + self.cfg.sigma[0] ** 2, device=self.device))  # Assume S == 1
        self.register_buffer("input_scale", input_scale)
        if self.cfg.get("meta", False):
            self.metaencoder = get_arch(cfg.metaencoder)
            self.metaout = nn.Linear(cfg.metaencoder.config.out_channels, 1, bias=False)
        else:
            self.metaencoder, self.metaout = None, None

    @property
    def sigma(self):
        return self.cfg.sigma[0]

    @staticmethod
    def finalize_config(cfg):
        with read_write(cfg):
            cfg.phi.config.in_channels = cfg.input_shape[0] * cfg.M[0]
            cfg.phi.config.out_channels = cfg.input_shape[0] * cfg.M[0]
            if cfg.get("meta", False) or cfg.get("metaencoder", None):
                cfg.metaencoder.config.in_channels = 2 * cfg.M[0] * cfg.input_shape[0]
        return cfg

    def forward(self, y: torch.Tensor):
        nu = self.phi(y / self.input_scale)
        energy = 0.5 * torch.pow(y - nu, 2).sum(dim=(1, 2, 3)) / self.sigma ** 2
        if self.metaencoder is not None:
            meta_input = torch.cat([y / self.input_scale, nu], dim=1)
            meta = self.metaout(torch.flatten(self.metaencoder(meta_input), start_dim=1))
            meta = 0.5 * meta.squeeze(-1).squeeze(-1) / self.sigma ** 2
            energy = energy + meta
        return energy

    def training_step(self, batch: torch.Tensor, batch_idx):
        x_0, _ = batch
        x_0 = x_0.permute(0, 3, 1, 2).contiguous().float() / 255.0 if self.cfg.dataset == "ffhq256" else x_0
        x = torch.cat(self.M * (x_0,), dim=1)  # expand channels from C to M * C with copies
        y = x + self.sigma * torch.randn_like(x)  # add a different noise measurement for every C channels

        loss = self.mem_loss(x, y)
        self.log("train_loss", loss.item(), on_step=True, sync_dist=self.sync_log)
        return loss

    def training_epoch_end(self, outputs) -> None:
        torch.cuda.empty_cache()

    def score(self, y: torch.Tensor):
        y.requires_grad_()
        with torch.set_grad_enabled(True):  # grad must be enabled whenever we want to compute score
            energy = torch.sum(self(y))
            score = torch.autograd.grad(-energy, y, create_graph=self.training)[0]
        return score

    def xhat(self, y: torch.Tensor):
        return y + self.score(y).mul(self.sigma ** 2)

    @torch.no_grad()
    def overdamped(self, y: torch.Tensor, v: torch.Tensor, options: dict):
        delta, steps = options["delta"], options["steps"]
        noise = torch.randn(steps, *y.shape)
        for i in range(steps):
            psi = self.score(y)
            y += pow(delta, 2) * psi / 2 + delta * noise[i].to(self.device)
            if (i % 250) == 0:
                torch.cuda.empty_cache()
        return y

    @torch.no_grad()
    def underdamped(self, y: torch.Tensor, v: torch.Tensor, options: dict):
        delta, gamma, lipschitz, steps = options["delta"], options["friction"], options["lipschitz"], options["steps"]
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(-gamma * delta)
        zeta2 = math.exp(-2 * gamma * delta)
        for i in range(steps):
            y += delta * v / 2  # y_{t+1}
            psi = self.score(y)
            v += u * delta * psi / 2  # v_{t+1}
            v = zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * torch.randn_like(y)  # v_{t+1}
            y += delta * v / 2  # y_{t+1}
            torch.cuda.empty_cache()
        return y, v

    @torch.no_grad()
    def chengetal(self, y: torch.Tensor, v: torch.Tensor, options: dict):
        delta, gamma, lipschitz, steps = options["delta"], options["friction"], options["lipschitz"], options["steps"]
        u = pow(lipschitz, -1)  # inverse mass
        gamma_ = pow(gamma, -1)
        zeta1 = math.exp(-gamma * delta)
        zeta2 = math.exp(-2 * gamma * delta)
        # construct the covariance matrix
        yy = u * gamma_ * (2 * delta - 4 * gamma_ * (1 - zeta1) + gamma_ * (1 - zeta2))
        vv = u * (1 - zeta2)
        yv = u * gamma_ * (1 - 2 * zeta1 + zeta2)
        # cholesky decomposition (upper=False)
        ch11 = pow(yy, 1 / 2)
        ch22 = pow(vv - pow(yv, 2) / yy, 1 / 2)
        ch21 = yv / pow(yy, 1 / 2)
        noise_B1 = torch.randn(steps, *y.shape)
        noise_B2 = torch.randn(steps, *y.shape)

        for i in range(steps):
            psi = self.score(y)
            # compute the conditional means of y and v
            y += gamma_ * (1 - zeta1) * v + u * gamma_ * (delta - gamma_ * (1 - zeta1)) * psi
            v.mul_(zeta1).add_(psi.mul_(u * gamma_ * (1 - zeta1)))  # v = v * zeta1 + u * gamma_ * (1 - zeta1) * psi
            # sample from the conditional gaussian using the cholesky decomposition
            y = y + ch11 * noise_B1[i].to(self.device)
            v = v + ch21 * noise_B1[i].to(self.device) + ch22 * noise_B2[i].to(self.device)
            if (i % 250) == 0:
                torch.cuda.empty_cache()

        return y, v

    def mem_loss(self, x, y):
        n, c = x.shape[0], self.input_shape[0]
        xhat = self.xhat(y)
        batch_errors = (x - xhat).pow(2).sum(dim=(2, 3))  # sum over pixels
        batch_errors_per_scale = batch_errors.view(n, self.M, c).sum(dim=2)  # sum on RGB channels
        batch_errors_mean = batch_errors_per_scale.mean(dim=0)  # mean over batches: size=(model.M)
        loss = batch_errors_mean.mean()
        return loss

    def configure_optimizers(self):
        return get_optimizer(self.parameters(), self.cfg.training.optimizer)
