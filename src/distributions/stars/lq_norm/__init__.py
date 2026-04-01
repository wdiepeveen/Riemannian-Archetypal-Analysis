import torch
import math

from src.distributions.stars import StarDistribution
from src.radials.unimodal.lq_norm import LqNormRadial

class LqNormStarDistribution(StarDistribution):
    def __init__(self, d, q, alpha=1.0):
        radial = LqNormRadial(d, q, alpha)
        super().__init__(d, radial)

    def _log_normalizing_constant(self):
        # C = alpha^d * (2 * pi)^(d/2) (2^(q/2) Gamma((q+1)/2)/ pi^(1/2))^(-d/q) * d^(-(1/q - 1/2) d)
        d = self.d
        q = self.radial.q
        alpha = self.radial.alpha
        log_C = - torch.tensor([d * math.log(alpha) + (d / 2) * math.log(2 * math.pi) - (d / q) * (math.log(2 ** (q / 2) * math.gamma((q + 1) / 2) / math.sqrt(math.pi))) - (1 / q - 0.5) * d * math.log(d)])
        return log_C
        