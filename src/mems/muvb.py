from typing import Tuple

import torch
from omegaconf import DictConfig, read_write

from .mem import MEM
from .mvae import MVAE


class MUVB(MEM):
    """
    Multimeasurement Un-normalized Variational Bayes.
    S := Number of noise scales == len(cfg.model.vae.sigma). We assume S == 1 here.
    M(s) := number of noisy measurements at s^{th} scale == M(cfg.model.vae.M[s]).
    Note that the energy (phi) is in Y space (y is a stack of noisy observations), so all methods expect NOISY data.
    The expected input shape is (batch, channels * T, height, width).
    Clean inputs (x_0) are only expected during training, by self.training_step().
    """

    def __init__(self, cfg: DictConfig):
        super(MEM, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.M = cfg.vae.M[0]  # Assume S = 1
        self.vae = MVAE(cfg=cfg.vae)
        self.input_shape = cfg.vae.input_shape
        self.sync_log = True if cfg.training.get("sync_log", None) is True else False
        self.grad_norms_prev_epoch, self.grad_norms_this_epoch = [], []
        self.example_input_array = torch.zeros(
            1, self.M * self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        self.score_norm = None

    @property
    def sigma(self):
        return self.vae.sigma

    @staticmethod
    def finalize_config(cfg):
        with read_write(cfg):
            cfg.vae = MVAE.finalize_config(cfg.vae)
        return cfg

    def forward(self, y: torch.Tensor):
        energy = -self.vae.elbo(y)
        return energy

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        x, y = self.vae.make_xy_from_x0(batch[0], self.cfg.training.yperx)
        y.requires_grad_()

        loss = self.mem_loss(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_log)
        return loss
