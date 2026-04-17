import torch

from radials.unimodal.centered_ellipsoid import CenteredEllipsoidRadial

class GaussianEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, cov, p=0.95):
        self.cov = cov
        self.p = p
        super().__init__(cov.shape[0])

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
        return torch.linalg.inv(r ** 2 * self.cov)