import torch

from src.radials.unimodal.elliposoid.offcentered import OffCenteredEllipsoidRadial

class GaussianEnclosingOffCenteredEllipsoidRadial(OffCenteredEllipsoidRadial):
    def __init__(self, cov, mu, c=4/3, p=0.95):
        self.cov = cov
        self.mu = mu
        self.p = p
        super().__init__(cov.shape[0], c=c)

    def normal_quantile(self, eps=1e-10):
        """
        Approximate Phi^{-1}(p) using erfinv; fully differentiable.
        p can be tensor or scalar in (0,1).
        """
        p = torch.as_tensor(self.p)
        p = torch.clamp(p, eps, 1 - eps)
        return 2.0 ** 0.5 * torch.erfinv(2.0 * p - 1.0)

    def gaussian_ellipsoid_radius_approx(self, wh=False):
        """
        Differentiable approximation of r such that
        P(X^T Sigma^{-1} X <= r^2) ~= p for X ~ N(0, Sigma) in R^d.

        d: scalar or tensor of dimensions (degrees of freedom)
        p: scalar or tensor of probabilities in (0,1)
        wh: if True, use Wilson–Hilferty; else simple normal approx.
        """
        d = torch.as_tensor(self.d, dtype=torch.get_default_dtype())
        z = self.normal_quantile().to(d.dtype)

        if not wh:
            # Simple normal approximation: chi2_{d,p} ≈ d + sqrt(2d)*z
            q = d + torch.sqrt(2.0 * d) * z
        else:
            # Wilson–Hilferty: chi2_{d,p} ≈ d (1 - 2/(9d) + z sqrt(2/(9d)))^3
            d_safe = torch.clamp(d, min=1e-6)
            term = 1.0 - 2.0 / (9.0 * d_safe) + z * torch.sqrt(2.0 / (9.0 * d_safe))
            q = d_safe * term.pow(3)

        q = torch.clamp(q, min=0.0)
        return torch.sqrt(q)
    
    def construct_Sigma_inv(self):
        r = self.gaussian_ellipsoid_radius_approx()
        lambda_1 = torch.maximum(self.c ** 2 * self.mu.norm(2) ** 2, r ** 2 * torch.einsum('i,ij,j->', self.mu, self.cov, self.mu) / torch.dot(self.mu, self.mu))
        mu_o_mu = torch.outer(self.mu, self.mu) / self.mu.norm(2) ** 2
        Sigma = lambda_1 * mu_o_mu + r ** 2 * (torch.eye(self.mu.shape[0]) - mu_o_mu) @ self.cov @ (torch.eye(self.mu.shape[0]) - mu_o_mu)
        return torch.linalg.inv(Sigma)