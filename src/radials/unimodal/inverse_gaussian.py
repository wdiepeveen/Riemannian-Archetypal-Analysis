import torch

from src.radials.unimodal import UniModalRadial

class InverseGaussianRadial(UniModalRadial):
    def __init__(self, d, mu, lam):
        """
        
        :param mu: d tensor
        :param lam: d tensor
        """
        super().__init__(d)

        self.mu = mu
        self.lam = lam  

    def forward(self, theta):
        return 1/torch.norm((theta - self.mu[None]) / self.lam[None], dim=-1) 