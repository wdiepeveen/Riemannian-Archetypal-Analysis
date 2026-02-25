import torch
import torch.nn as nn

from src.transforms.vector.linear import LinearVectorTransform

class StructuredBlockLowerTriangularVectorTransform(LinearVectorTransform):
    def __init__(self, d, r, bias=False, efficient_inverse=False, random_init=False):
        super().__init__(d)
        self.r = r

        if random_init: # how one would normally initalize linear layers (scaled properly)
            # intialize as r x d matrix and extract the relevant blocks
            init_matrix = torch.nn.init.kaiming_uniform_(torch.empty(d, r), a=0, mode='fan_in', nonlinearity='linear')
            init_rr = init_matrix[:r, :]
            init_dr = init_matrix[r:, :]
        else:
            init_rr = torch.eye(r)
            init_dr = torch.zeros(d - r, r)

        self.weight_11 = nn.Parameter(init_rr)
        self.weight_21 = nn.Parameter(init_dr)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.d))
        else:
            # Register as buffer so it moves with .to(), .cuda(), etc.
            self.register_buffer("bias", torch.zeros(self.d))

        self.efficient_inverse = efficient_inverse

    def forward(self, x, context=None):
        if self.efficient_inverse:
            return self._inverse(x, context=context)
        else:
            return self._forward(x, context=context)
        
    def inverse(self, z, context=None):
        if self.efficient_inverse:
            return self._forward(z, context=context)
        else:
            return self._inverse(z, context=context)
        
    def _forward(self, x, context=None):
        x1, x2 = x[:, :self.r], x[:, self.r:]
        z1 = x1 @ self.weight_11.t()
        z2 = x1 @ self.weight_21.t() + x2
        z = torch.cat([z1, z2], dim=1) + self.bias
        log_abs_det = torch.det(self.weight_11).abs().log() * torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        return z, log_abs_det

    def _inverse(self, z, context=None):
        z1, z2 = z[:, :self.r], z[:, self.r:]
        x1 = z1 @ torch.inverse(self.weight_11).t()
        x2 = (z2 - x1 @ self.weight_21.t())
        x = torch.cat([x1, x2], dim=1) - self.bias
        log_abs_det = -torch.det(self.weight_11).abs().log() * torch.ones(z.size(0), device=z.device, dtype=z.dtype)
        return x, log_abs_det