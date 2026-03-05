import torch

from src.radials.unimodal.inverse_affine_norm import InverseAffineNormRadial

class InverseLinearNormRadial(InverseAffineNormRadial):
    def __init__(self, d, r=None, sigma=1e-3):
        """
        
        :param centered: bool, whether the linear layer should have a bias term
        """
        super().__init__(d, sigma=sigma)
        self.r = r if r is not None else d

        self.linear = torch.nn.Linear(self.d, self.r, bias=False)

    def affine(self, theta):
        return self.linear(theta)
        
    @property
    def mode_variance(self):
        """
        Return the variance of the mode of the radial.
        """
        with torch.no_grad():
            W = self.linear.weight.clone()
            return torch.inverse(W.t() @ W)