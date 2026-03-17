import torch

from src.radials.unimodal.elliposoid import EllipsoidRadial

class OffCenteredEllipsoidRadial(EllipsoidRadial):
    def __init__(self, cov, mu, c=4/3):
        self.cov = cov
        self.mu = mu
        self.c = c
        super().__init__(d=cov.shape[0])

    def compute_intersect(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        a = torch.einsum("ij,jk,ik->i", theta, self.Sigma_inv.to(theta.dtype), theta)
        b = -2 * torch.einsum("ij,jk,k->i", theta, self.Sigma_inv.to(theta.dtype), self.mu.to(theta.dtype))
        c = torch.einsum("i,i->", self.mu.to(theta.dtype), self.Sigma_inv.to(theta.dtype) @ self.mu.to(theta.dtype)) - 1
        discriminant = b ** 2 - 4 * a * c
        assert (discriminant >= 0).all(), "discriminant must be non-negative"
        t = (-b + torch.sqrt(discriminant)) / (2 * a)
        assert (t >= 0).all(), "t must be non-negative"
        return t

    def construct_Sigma(self):
        raise NotImplementedError("construct_Sigma must be implemented by subclasses")