import torch

from .mem import MEM


class MDAE(MEM):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.cfg.get("sigmoid", True):
            nu = self.phi(y / self.input_scale).sigmoid()
        else:
            nu = self.phi(y / self.input_scale)
        score = (nu - y) / (self.sigma ** 2)
        return score

    def score(self, y: torch.Tensor) -> torch.Tensor:
        return self(y)
