import torch

from src.radials.unimodal.elliposoid.offcentered import OffCenteredEllipsoidRadial

class EllipsoidEnclosingOffCenteredEllipsoidRadial(OffCenteredEllipsoidRadial):
    def __init__(self, cov, mu, c=4/3):
        super().__init__(cov, mu, c=c)

    def construct_Sigma(self):
        # torch.einsum('i,ij,j->', mu, Sigma_trim, mu) / torch.dot(mu, mu)
        lambda_1 = torch.maximum(self.c ** 2 * self.mu.norm(2) ** 2, torch.einsum('i,ij,j->', self.mu, self.cov, self.mu) / torch.dot(self.mu, self.mu))
        mu_o_mu = torch.outer(self.mu, self.mu) / self.mu.norm(2) ** 2
        Sigma = lambda_1 * mu_o_mu + (torch.eye(self.mu.shape[0]) - mu_o_mu) @ self.cov @ (torch.eye(self.mu.shape[0]) - mu_o_mu)
        return Sigma