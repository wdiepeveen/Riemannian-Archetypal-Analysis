import torch 

class Concave(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r):
        """
        r: N
        returns v(r): N
        """
        raise NotImplementedError