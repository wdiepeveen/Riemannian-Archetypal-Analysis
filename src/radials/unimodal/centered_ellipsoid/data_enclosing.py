import torch

from src.radials.unimodal.centered_ellipsoid import CenteredEllipsoidRadial

class DataEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, data, center, c=1.1, reg_param=1e-2):
        self.m = data
        self.mu = center
        self.reg_param = reg_param

        self.c = c

        super().__init__(data.shape[1])

    def construct_Sigma_inv(self):
        _U, sig, _ = torch.linalg.svd((torch.eye(self.d) - torch.outer(self.mu, self.mu)/ torch.norm(self.mu)**2) @ self.m.T)
        # pick r to be the number of singular values above a threshold
        r = (sig > self.reg_param).sum().item()
        U = torch.cat([self.mu[:, None] / torch.norm(self.mu), (torch.eye(self.d) - torch.outer(self.mu, self.mu)/ torch.norm(self.mu)**2) @ _U[:, :r]], dim=1)
        sq_m_norms = (self.m @ U) ** 2
        sq_mu_norm = ((self.mu[None]) @ U) ** 2
        sq_norms = torch.cat([sq_m_norms, sq_mu_norm], dim=0)
        lambda_ = (r+1) * self.c * torch.max(sq_norms, dim=0).values
        Sigma_inv = 1 / self.reg_param * (torch.eye(self.d) - U @ U.T) + U @ torch.diag(1/(lambda_ + self.reg_param)) @ U.T
        return Sigma_inv
