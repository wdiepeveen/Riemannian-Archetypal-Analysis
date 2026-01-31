import torch
import torch.nn as nn

from src.transforms.vector.linear import LinearVectorTransform


class LowerTriangularLinearVectorTransform(LinearVectorTransform):
    def __init__(self, d, bias=False, efficient_inverse=False):
        super().__init__(d)

        # Strictly lower-triangular part is learnable; diagonal is fixed to 1
        init = torch.tril(torch.zeros(self.d, self.d), diagonal=-1)
        self.weight = nn.Parameter(init)

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

    def _weight_matrix(self, x):
        # Construct full lower-triangular matrix with ones on the diagonal
        eye = torch.eye(self.d, device=x.device, dtype=x.dtype)
        return torch.tril(self.weight, diagonal=-1) + eye

    def _forward(self, x, context=None):
        W = self._weight_matrix(x)
        z = x @ W.t()
        z = z + self.bias
        # Unit determinant: log|det| = 0
        log_abs_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return z, log_abs_det

    def _inverse(self, z, context=None):
        W = self._weight_matrix(z)
        x = z - self.bias
        x = x @ torch.inverse(W).t()
        log_abs_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        return x, log_abs_det
