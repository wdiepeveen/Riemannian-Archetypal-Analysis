import torch
from src.transforms.vector import VectorTransform


class RadialTransform(VectorTransform):
    def __init__(self, d):
        super().__init__(d)

    # def forward(self, x, context=None):
    #     """
        
    #     :param x: N x d tensor
    #     :return: z = x / rho(x/||x||) if ||x|| > 0 else 0
    #     """
    #     r = torch.linalg.norm(x, dim=-1)   
    #     direction = x / r[:,None]
    #     rho_x = self.radius(direction)              
    #     z = x / rho_x[:,None]
    #     return z,  - self.d * torch.log(rho_x)

    def forward(self, x, context=None):
        # forward: x -> z = x / rho(theta)
        eps = 1e-12
        r = torch.linalg.norm(x, dim=-1, keepdim=True)  # (..., 1)
        # safe directions
        theta = torch.where(
            r > eps, x / r, torch.zeros_like(x)
        )
        rho_theta = self.radius(theta)    
        # scale
        z = torch.where(
            r > eps, x / rho_theta[:,None], torch.zeros_like(x)
        )
        # log|det J| = -d * log rho(theta) + optional angular term
        logabsdet = self.d * torch.log(rho_theta)
        return z, logabsdet

    def inverse(self, z, context=None):
        # inverse: z -> x = rho(theta) * z
        eps = 1e-12
        r = torch.linalg.norm(z, dim=-1, keepdim=True)
        theta = torch.where(
            r > eps, z / r, torch.zeros_like(z)
        )
        rho_theta = self.radius(theta)
        x = torch.where(
            r > eps, rho_theta[:,None] * z, torch.zeros_like(z)
        )
        logabsdet = self.d * torch.log(rho_theta)
        return x, logabsdet

    def radius(self, theta):
        """
        :param theta: N x d tensor with ||theta|| = 1
        :return: N tensor
        """
        raise NotImplementedError
