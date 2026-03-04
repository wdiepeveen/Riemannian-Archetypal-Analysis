import torch

from nflows.distributions import Distribution

class DiagonalGaussian(Distribution):
    def __init__(self, d, sigma=1.0):
        super().__init__()
        self.d = d
        self.register_buffer("sigma", torch.as_tensor(sigma))

    def log_prob(self, inputs, context=None):
        return -0.5 * ((inputs / self.sigma) ** 2).norm(dim=-1) + self.d/2 * torch.log(2 * torch.pi * self.sigma ** 2)
    
    def sample(self, num_samples, context=None):
        return self.sigma * torch.randn(num_samples, self.d)
