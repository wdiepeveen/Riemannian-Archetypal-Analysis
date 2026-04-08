import torch
from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.vector import VectorDiffeomorphism

class StarVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, star_distribution):
        super().__init__(d)
        assert star_distribution.d == d, "Distribution dimension must match diffeomorphism dimension."
        self.rho = star_distribution.radial

    def forward(self, x):
        """
        Apply the radial transformation to x.
        
        :param x: N x d
        :return: N x d
        """
        eps = 1e-6
        r = torch.linalg.norm(x, dim=-1)              

        # safe theta
        theta = torch.zeros_like(x)
        theta[r > eps] = x[r > eps] / r[r > eps][..., None]                 
        
        rho_theta = torch.zeros_like(r)
        rho_theta[r > eps] = self.rho(theta[r > eps])               

        star_radius = r / (rho_theta + eps)        
        return theta * star_radius[..., None]  
    
    def inverse(self, y):
        """
        Apply the inverse of the radial transformation to y.
        
        :param y: N x d
        :return: N x d
        """
        eps = 1e-6
        r = torch.linalg.norm(y, dim=-1)         

        # safe theta
        theta = torch.zeros_like(y)
        theta[r > eps] = y[r > eps] / r[r > eps][..., None]                 
        
        rho_theta = torch.zeros_like(r)
        rho_theta[r > eps] = self.rho(theta[r > eps])

        star_radius = r * (rho_theta + eps)        
        return theta * star_radius[..., None]  

    def differential_forward(self, x, X):
        # J_phi(x) @ X, with graph kept
        _, Y = jvp(
            self.forward,
            (x,),
            (X,),
            create_graph=True,
            strict=True,
        )
        is_zero = (x.norm(dim=-1, keepdim=True) == 0)
        I_X = X
        return torch.where(is_zero, I_X, Y)

    def differential_inverse(self, y, Y):
        # J_phi^{-1}(y) @ Y
        _, X = jvp(
            self.inverse,
            (y,),
            (Y,),
            create_graph=True,
            strict=True,
        )
        is_zero = (y.norm(dim=-1, keepdim=True) == 0)
    # I_Y = Y
        return torch.where(is_zero, Y, X)

    def adjoint_differential_forward(self, x, X):
        # Adjoint is a VJP wrt x
        _, (adj_x,) = vjp(
            self.forward,
            (x,),
            (X,),
            create_graph=True,
            strict=True,
        )
        is_zero = (x.norm(dim=-1, keepdim=True) == 0)
        # At 0, use identity adjoint
        return torch.where(is_zero, X, adj_x)

    def adjoint_differential_inverse(self, y, Y):
        _, (adj_y,) = vjp(
            self.inverse,
            (y,),
            (Y,),
            create_graph=True,
            strict=True,
        )
        is_zero = (y.norm(dim=-1, keepdim=True) == 0)
        return torch.where(is_zero, Y, adj_y)    
