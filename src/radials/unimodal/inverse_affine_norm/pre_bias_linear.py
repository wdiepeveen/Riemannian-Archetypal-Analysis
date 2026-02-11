import torch

from src.radials.unimodal.inverse_affine_norm import InverseAffineNormRadial
from src.nn.module.pre_bias_linear import PreBiasLinear

class InversePreBiasLinearNormRadial(InverseAffineNormRadial):
    def __init__(self, d, centered=True):
        """
        
        :param centered: bool, whether the linear layer should have a bias term
        """
        super().__init__(d)

        self.pre_bias_linear = PreBiasLinear(self.d, self.d, bias=not centered)

    def affine(self, theta):
        return self.pre_bias_linear(theta)
    
    @property
    def mode_mean(self):
        """
        Return the mean of the mode of the radial.
        """
        if self.pre_bias_linear.bias is not None:
            with torch.no_grad():
                return self.pre_bias_linear.bias.clone()
        else:
            return torch.zeros(self.d)
        
    @property
    def mode_variance(self):
        """
        Return the variance of the mode of the radial.
        """
        with torch.no_grad():
            W = self.pre_bias_linear.weight.clone()
            return torch.inverse(W.t() @ W)