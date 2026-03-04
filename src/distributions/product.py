import torch
from torch import nn
from nflows.distributions import Distribution

class ProductDistribution(Distribution):
    def __init__(self, distributions):
        super().__init__()
        # Register as modules so their parameters are seen by .parameters()
        self.distributions = nn.ModuleList(distributions)
        self.d = sum(dist.d for dist in self.distributions)  # total dimension is sum of component dimensions

    def log_prob(self, x, context=None):
        dims = [d.d for d in self.distributions]
        cum = [0]
        for d in dims:
            cum.append(cum[-1] + d)

        log_probs = [
            dist.log_prob(x[:, cum[j]:cum[j+1]], context=context)
            for j, dist in enumerate(self.distributions)
        ]
        return sum(log_probs)


    def sample(self, num_samples, context=None):
        # each component samples over *all* features here;
        # adapt if you want a split over dimensions instead
        samples = [
            dist.sample(num_samples, context=context) 
            for dist in self.distributions
        ]
        return torch.cat(samples, dim=-1)
