import torch

from radials.unimodal.offcentered_ellipsoid import OffCenteredEllipsoidRadial

class EllipsoidEnclosingOffCenteredEllipsoidRadial(OffCenteredEllipsoidRadial):
    def __init__(self, cov, mu, c=4/3):
        self.cov = cov
        self.mu = mu
        super().__init__(cov.shape[0], c=c)

    def construct_Sigma_inv(self):
        lambda_1 = torch.maximum(self.c ** 2 * self.mu.norm(2) ** 2, torch.einsum('i,ij,j->', self.mu, self.cov, self.mu) / torch.dot(self.mu, self.mu))
        mu_o_mu = torch.outer(self.mu, self.mu) / self.mu.norm(2) ** 2
        Sigma = lambda_1 * mu_o_mu + (torch.eye(self.mu.shape[0]) - mu_o_mu) @ self.cov @ (torch.eye(self.mu.shape[0]) - mu_o_mu)
        return torch.linalg.inv(Sigma)