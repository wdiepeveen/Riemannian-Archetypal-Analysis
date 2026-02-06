import torch
from torch.autograd.functional import jvp

from src.diffeomorphisms.vector import VectorDiffeomorphism

class RadialVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, rho):
        super().__init__(d)
        self.rho = rho

    def forward(self, x):
        """
        Apply the radial transformation to x.
        
        :param x: N x d
        :return: N x d
        """
        eps = 1e-12
        r = torch.linalg.norm(x, dim=-1)              # (...,)

        # safe theta
        theta = torch.where(
            (r > eps)[..., None],
            x / (r + eps)[..., None],
            torch.zeros_like(x),
        )                                             # (..., d)

        rho_theta = self.rho(theta)                   # (...,)

        star_radius = r / (rho_theta + eps)           # (...)
        return theta * star_radius[..., None]         # (..., d)
    
    def inverse(self, y):
        """
        Apply the inverse of the radial transformation to y.
        
        :param y: N x d
        :return: N x d
        """
        eps = 1e-12
        r = torch.linalg.norm(y, dim=-1)              # (...,)

        # safe theta
        theta = torch.where(
            (r > eps)[..., None],
            y / (r + eps)[..., None],
            torch.zeros_like(y),
        )                                             # (..., d)

        rho_theta = self.rho(theta)                   # (...,)

        star_radius = r * (rho_theta + eps)           # (...)
        return theta * star_radius[..., None]         # (..., d)
    
    def differential_forward(self, x, X): # TODO if we compute differential at x = 0 we return identity
        return jvp(self.forward, (x,), (X,))[1]
    
    def differential_inverse(self, y, Y): # TODO if we compute differential at y = 0 we return identity
        return jvp(self.inverse, (y,), (Y,))[1]