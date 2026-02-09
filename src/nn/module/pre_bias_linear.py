import torch
import torch.nn as nn

class PreBiasLinear(nn.Module):
    """
    Like nn.Linear, but applies bias to the input first:
        y = (x - b_in) @ W^T
    where b_in has shape (in_features,) and W has shape (out_features, in_features).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Same convention as nn.Linear: W shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # You can mirror nn.Linear's init if you like; for simplicity use kaiming_uniform-like.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            # small uniform init
            fan_in = self.in_features
            bound = 1.0 / fan_in**0.5 if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        if self.bias is not None:
            x = x - self.bias  # broadcasts over leading dims
        return x @ self.weight.t()
