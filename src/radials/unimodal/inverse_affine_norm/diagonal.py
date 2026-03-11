import torch
import torch.nn as nn

from src.radials.unimodal.inverse_affine_norm import InverseAffineNormRadial

class InverseDiagonalNormRadial(InverseAffineNormRadial):
    def __init__(self, d, diagonal=None sigma=1e-6):
        """
        
        :param centered: bool, whether the linear layer should have a bias term
        """
        super().__init__(d, sigma=sigma)
        if diagonal is None:
            self.diagonal = nn.Parameter(torch.randn(self.d).exp())
            

    def affine(self, theta):
        return self.diagonal * theta
        
    @property
    def mode_variance(self):
        """
        Return the variance of the mode of the radial.
        """
        with torch.no_grad():
            return torch.diag(1.0 / (self.diagonal ** 2))