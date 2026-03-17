import torch

from nflows.distributions import Distribution
from nflows.distributions.normal import DiagonalNormal

class DiagonalGaussian(Distribution):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.diagonal_normal = DiagonalNormal([self.d])

    def _log_prob(self, inputs, context=None):
        return self.diagonal_normal._log_prob(inputs, context)
    
    # def sample(self, num_samples, context=None):
    #     return self.diagonal_normal.sample(num_samples, context)
    
    def _sample(self, num_samples, context):
        means = self.diagonal_normal.mean_                    # shape (1, d)
        log_stds = self.diagonal_normal.log_std_             # shape (1, d)
        stds = torch.exp(log_stds)

        noise = torch.randn(num_samples, *self.diagonal_normal._shape, device=means.device)
        samples = means + stds * noise       # broadcasting over batch
        return samples
