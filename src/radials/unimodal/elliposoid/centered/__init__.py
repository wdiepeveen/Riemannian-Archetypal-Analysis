import torch

from src.radials.unimodal.elliposoid import EllipsoidRadial

class CenteredEllipsoidRadial(EllipsoidRadial):
    def __init__(self, cov):
        self.cov = cov
        super().__init__(d=cov.shape[0])

    def compute_intersect(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        a = torch.einsum("ij,jk,ik->i", theta, self.Sigma_inv.to(theta.dtype).to(theta.device), theta)
        discriminant =  4 * a 
        assert (discriminant >= 0).all(), "discriminant must be non-negative"
        t = (torch.sqrt(discriminant)) / (2 * a)
        assert (t >= 0).all(), "t must be non-negative"
        return t

    def construct_Sigma(self):
        raise NotImplementedError("construct_Sigma must be implemented by subclasses")