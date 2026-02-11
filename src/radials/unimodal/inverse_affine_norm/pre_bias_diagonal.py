import torch
import torch.nn as nn

from src.radials.unimodal.inverse_affine_norm import InverseAffineNormRadial
from src.nn.module.pre_bias_linear import PreBiasLinear

class InversePreBiasDiagonalNormRadial(InverseAffineNormRadial):
    def __init__(self, d, centered=True):
        """
        
        :param centered: bool, whether the linear layer should have a bias term
        """
        super().__init__(d)

        self.diagonal = nn.Parameter(torch.randn(self.d).exp())
        if not centered:
            self.bias = nn.Parameter(torch.zeros(self.d))
        else:
            self.register_parameter('bias', None)

    def affine(self, theta):
        if self.bias is not None:
            return self.diagonal * (theta - self.bias)
        else:
            return self.diagonal * theta
    
    @property
    def mode_mean(self):
        """
        Return the mean of the mode of the radial.
        """
        if self.bias is not None:
            with torch.no_grad():
                return self.bias.clone()
        else:
            return torch.zeros(self.d)
        
    @property
    def mode_variance(self):
        """
        Return the variance of the mode of the radial.
        """
        with torch.no_grad():
            return torch.diag(1.0 / (self.diagonal ** 2))