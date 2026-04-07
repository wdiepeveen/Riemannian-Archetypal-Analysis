import torch
from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.vector import VectorDiffeomorphism

class ConcaveVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, concave):
        super().__init__(d)
        self.v = concave

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

        v_r = self.v(r)                          
        return theta * v_r[..., None]  
    
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

        v_r = self.v.inverse(r)                          
        return theta * v_r[..., None]
    
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