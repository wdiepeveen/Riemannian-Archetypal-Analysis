import torch 

class Radial(torch.nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.d = d

    def forward(self, theta):
        """
        theta: N x d unit vectors
        returns rho(theta): N
        """
        raise NotImplementedError