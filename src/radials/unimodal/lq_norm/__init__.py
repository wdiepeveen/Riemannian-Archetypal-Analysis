import torch
from src.radials.unimodal import UniModalRadial

class LqNormRadial(UniModalRadial):
    def __init__(self, d, q, alpha=1.0):
        super().__init__(d)
        self.q = q
        self.alpha = alpha

    def forward(self, theta):
        return self.alpha / torch.linalg.vector_norm(theta, ord=self.q, dim=-1)
        # return self.alpha * (abs(theta) ** self.q).sum(-1) ** (-1 / self.q)