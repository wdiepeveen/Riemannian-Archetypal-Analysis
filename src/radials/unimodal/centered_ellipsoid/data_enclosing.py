import torch

from src.radials.unimodal.centered_ellipsoid import CenteredEllipsoidRadial

class DataEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, data, center, alpha=1.1, beta=1e-2):
        self.m = data.shape[0]
        self.y = data
        self.mu = center
        self.alpha = alpha
        self.beta = beta

        super().__init__(data.shape[1])

    def construct_Sigma_inv(self):
        _U, sig, _ = torch.linalg.svd((torch.eye(self.d) - torch.outer(self.mu, self.mu) / torch.norm(self.mu)**2) @ self.y.T)
        # pick q to be the number of singular values above a threshold
        q = ((self.d * sig**2 / self.m) >= self.beta).sum().item()
        # construct U using the top q singular vectors and the direction of mu
        U = torch.cat([self.mu[:, None] / torch.norm(self.mu), (torch.eye(self.d) - torch.outer(self.mu, self.mu)/ torch.norm(self.mu)**2) @ _U[:, :q]], dim=1)
        # construct lambdas using the top q singular values and the direction of mu
        lambdas = torch.zeros(q+1)
        lambda_0 = self.d * torch.sum((self.mu[None] * self.y) ** 2 / torch.norm(self.mu)**2) / self.m
        lambdas[0] = lambda_0 if lambda_0 > self.alpha else self.alpha
        lambdas[1:] = self.d * sig[:q]**2 / self.m

        Sigma_inv = 1 / self.beta * (torch.eye(self.d) - U @ U.T) + U @ torch.diag(1 / lambdas) @ U.T
        return Sigma_inv
