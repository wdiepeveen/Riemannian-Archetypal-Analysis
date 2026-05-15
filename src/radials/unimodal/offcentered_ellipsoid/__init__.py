import torch

from src.radials.unimodal import UniModalRadial

class OffCenteredEllipsoidRadial(UniModalRadial):
    def __init__(self, d):
        super().__init__(d)
        self.Sigma_inv = self.construct_Sigma_inv()

    def forward(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        return self.compute_intersect(theta)

    def compute_intersect(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        a = torch.einsum("ij,jk,ik->i", theta, self.Sigma_inv.to(theta.dtype).to(theta.device), theta)
        b = -2 * torch.einsum("ij,jk,k->i", theta, self.Sigma_inv.to(theta.dtype).to(theta.device), self.mu.to(theta.dtype).to(theta.device))
        c = torch.einsum("i,i->", self.mu.to(theta.dtype).to(theta.device), self.Sigma_inv.to(theta.dtype).to(theta.device) @ self.mu.to(theta.dtype).to(theta.device)) - 1
        discriminant = b ** 2 - 4 * a * c
        assert (discriminant >= 0).all(), "discriminant must be non-negative"
        t = (-b + torch.sqrt(discriminant)) / (2 * a)
        assert (t >= 0).all(), "t must be non-negative"
        return t

    def construct_Sigma_inv(self):
        raise NotImplementedError("construct_Sigma_inv must be implemented by subclasses")