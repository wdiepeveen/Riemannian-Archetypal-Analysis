import torch
from torch.autograd.functional import jvp

from src.diffeomorphisms.vector import VectorDiffeomorphism

class StarGaussianVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, star_gaussian_distribution):
        super().__init__(d)
        assert star_gaussian_distribution.d == d, "Distribution dimension must match diffeomorphism dimension."
        self.rho = star_gaussian_distribution.radial

    def forward(self, x):
        """
        Apply the radial transformation to x.
        
        :param x: N x d
        :return: N x d
        """
        eps = 1e-12
        r = torch.linalg.norm(x, dim=-1)              

        # safe theta
        theta = torch.where(
            (r > eps)[..., None],
            x / (r + eps)[..., None],
            torch.zeros_like(x),
        )                                         

        rho_theta = self.rho(theta)                   

        star_radius = r / (rho_theta + eps)        
        return theta * star_radius[..., None]  
    
    def inverse(self, y):
        """
        Apply the inverse of the radial transformation to y.
        
        :param y: N x d
        :return: N x d
        """
        eps = 1e-12
        r = torch.linalg.norm(y, dim=-1)         

        # safe theta
        theta = torch.where(
            (r > eps)[..., None],
            y / (r + eps)[..., None],
            torch.zeros_like(y),
        )                                            

        rho_theta = self.rho(theta)              

        star_radius = r * (rho_theta + eps)        
        return theta * star_radius[..., None]      

    def differential_forward(self, x, X):
        """
        Apply the differential map of the radial transformation at x for a vector X.

        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        Y = jvp(self.forward, (x,), (X,))[1]
        is_zero = (x.norm(dim=-1, keepdim=True) == 0)
        I_X = X  # differential of identity at 0 applied to X is just X
        return torch.where(is_zero, I_X, Y)

    def differential_inverse(self, y, Y):
        """
        Apply the differential map of the inverse of the radial transformation at y for a vector Y.

        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        X = jvp(self.inverse, (y,), (Y,))[1]
        is_zero = (y.norm(dim=-1, keepdim=True) == 0)
        I_Y = Y  # differential of identity at 0 applied to Y is just Y
        return torch.where(is_zero, I_Y, X)