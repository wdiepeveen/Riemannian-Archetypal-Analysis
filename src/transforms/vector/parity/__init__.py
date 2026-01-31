import torch

from src.transforms.vector import VectorTransform

class ParityVectorTransform(VectorTransform):
    def __init__(self, d, r, network, parity):
        super().__init__(d)

        self.r = r
        self.nn = network
        self.parity = parity % 2


    def forward(self, x, context=None):
        log_abs_det = torch.zeros(1, device=x.device)
        
        # Apply non-linearity
        z = torch.zeros_like(x)
        if self.parity == 0:
            z[:,:self.r] = x[:,:self.r]
            z[:,self.r:] = x[:,self.r:] + self.nn(x[:,:self.r])
        else:
            z[:,(self.d - self.r):] = x[:,(self.d - self.r):]
            z[:,:(self.d - self.r)] = x[:,:(self.d - self.r)] + self.nn(x[:,(self.d - self.r):])

        return z, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        log_abs_det = torch.zeros(1, device=z.device)
    
        # Apply non-linearity
        x = torch.zeros_like(z)
        if self.parity == 0:
            x[:,:self.r] = z[:,:self.r]
            x[:,self.r:] = z[:,self.r:] - self.nn(z[:,:self.r])
        else:
            x[:,(self.d - self.r):] = z[:,(self.d - self.r):]
            x[:,:(self.d - self.r)] = z[:,:(self.d - self.r)] - self.nn(z[:,(self.d - self.r):])

        return x, log_abs_det.expand(z.shape[0])