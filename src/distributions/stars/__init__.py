import math
import torch
from nflows.distributions.base import Distribution

class StarDistribution(Distribution):
    r"""
    Star-Gaussian in R^d with density

        p(x) = C * exp(-0.5 * (||x|| / rho(theta))^2),

    where x = r * theta, r > 0, theta = x / ||x||, and rho(theta) > 0 is trainable.
    The normalizing constant C is approximated by a differentiable MC integral
    over the sphere.
    """

    def __init__(self, d, radial, n_mc_norm=2048, mcmc_steps=10, mcmc_step_size=0.1):
        super().__init__()
        self.d = d
        self.radial = radial
        assert self.d == radial.d

        self.n_mc_norm = n_mc_norm
        self.mcmc_steps = mcmc_steps
        self.mcmc_step_size = mcmc_step_size

        # dummy scalar buffer, only used to carry device/dtype
        self.register_buffer(
            "_const_buffer",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )

    def rho(self, theta):
        """
        theta: (..., d) unit vectors
        returns rho(theta): (...,)
        """
        return self.radial(theta)

    # ---------- log normalizing constant (differentiable MC) ----------

    def _log_normalizing_constant(self, ):
        """
        Estimate log C where

            C = 1 / ( 2^{d/2 - 1} Gamma(d/2) * ∫_S rho(ω)^d dσ(ω) ).

        We approximate the sphere integral by MC with uniform samples on S^{d-1}
        obtained via reparameterization (Gaussian -> normalize).
        """
        d = self.d
        n_mc = self.n_mc_norm

        # 1. sample omega ~ Unif(S^{d-1}) by normalizing Gaussians (reparameterized)
        eps = torch.randn(n_mc, d, device=self._const_buffer.device)
        omega = eps / eps.norm(dim=-1, keepdim=True)  # (n_mc, d)

        # 2. compute rho(omega)^d with gradients
        rho_omega = self.rho(omega)                   # (n_mc,)
        rho_pow = rho_omega.pow(d)                    # (n_mc,)

        # 3. MC estimate of I(psi) = ∫_S rho(ω)^d dσ(ω)
        area = 2.0 * math.pi ** (d / 2.0) / math.gamma(d / 2.0)
        I_hat = area * rho_pow.mean()                 # scalar, differentiable

        const = 2.0 ** (d / 2.0 - 1.0) * math.gamma(d / 2.0)
        C_hat = 1.0 / (const * I_hat)
        return torch.log(C_hat)

    # ---------- log_prob ----------

    def _log_prob(self, x, context=None):
        """
        x: (..., d)
        log p(x) = log C - 0.5 * (||x|| / rho(theta))^2
        with theta = x / ||x||.
        """
        eps = 1e-12
        x = x.to(self._const_buffer.device)
        r = torch.linalg.norm(x, dim=-1)              # (...,)

        # safe theta
        theta = torch.where(
            (r > eps)[..., None],
            x / (r + eps)[..., None],
            torch.zeros_like(x),
        )                                             # (..., d)

        rho_theta = self.rho(theta)                   # (...,)

        star_radius = r / (rho_theta + eps)           # (...)
        log_unnormalized = -0.5 * star_radius**2      # (...)

        logC = self._log_normalizing_constant()       # scalar
        return logC + log_unnormalized

    # ---------- sampling ----------

    def _logw(self, theta):
        """
        log weight for theta ~ rho(theta)^d.
        theta: (N, d) on S^{d-1}
        """
        return self.d * torch.log(self.rho(theta))  # (N,)

    @torch.no_grad()
    def _sample_theta(self, num_samples):
        """
        Crude random-walk Metropolis on S^{d-1} with target density ∝ rho(theta)^d.
        Returns (num_samples, d) unit vectors.
        """
        d = self.d

        # initialize from uniform on S^{d-1}
        theta = torch.randn(num_samples, d, device=self._const_buffer.device)
        theta = theta / theta.norm(dim=-1, keepdim=True)

        logw_theta = self._logw(theta)

        for _ in range(self.mcmc_steps):
            prop = theta + self.mcmc_step_size * torch.randn_like(theta)
            prop = prop / prop.norm(dim=-1, keepdim=True)
            logw_prop = self._logw(prop)

            log_alpha = logw_prop - logw_theta
            u = torch.rand_like(log_alpha)
            accept = (torch.log(u) < log_alpha).float()[..., None]  # (N,1)

            theta = accept * prop + (1.0 - accept) * theta
            logw_theta = accept.squeeze(-1) * logw_prop + (1.0 - accept.squeeze(-1)) * logw_theta

        return theta

    def _sample(self, num_samples, context=None):
        """
        Sampling algorithm:

        1. theta ~ density ∝ rho(theta)^d on S^{d-1} (MCMC).
        2. U ~ Gamma(d/2, 1), R = rho(theta) * sqrt(2U).
        3. X = R * theta.
        """
        theta = self._sample_theta(num_samples)            # (N, d)

        # Gamma(shape=d/2, rate=1)
        gamma_dist = torch.distributions.Gamma(self.d / 2.0, 1.0)
        U = gamma_dist.sample((num_samples,)).to(self._const_buffer.device)   # (N,)

        rho_theta = self.rho(theta)                        # (N,)
        R = rho_theta * (2.0 * U).sqrt()                   # (N,)

        x = R[:, None] * theta                             # (N, d)
        return x
