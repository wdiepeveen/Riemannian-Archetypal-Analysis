import torch

from src.transforms.vector import VectorTransform

class ParityVectorTransform(VectorTransform):
    def __init__(self, d, r, network, parity):
        super().__init__(d)

        self.r = r
        self.nn = network
        self.parity = parity % 2

        self.mask = self.generate_vector_mask()


    def forward(self, x, context=None):
        log_abs_det = torch.zeros(1, device=x.device)
        
        # Apply non-linearity
        z = torch.zeros_like(x) 
        z[:,self.mask] = x[:,self.mask]
        z[:,~self.mask] = x[:,~self.mask] + self.nn(x[:,self.mask])

        return z, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        log_abs_det = torch.zeros(1, device=z.device)
    
        # Apply non-linearity
        x = torch.zeros_like(z)
        x[:,self.mask] = z[:,self.mask]
        x[:,~self.mask] = z[:,~self.mask] - self.nn(z[:,self.mask])

        return x, log_abs_det.expand(z.shape[0])
    
    def generate_vector_mask(self):
        mask = torch.zeros(self.d, dtype=torch.bool)
        if self.parity == 0:
            mask[:self.r] = 1
        else:
            mask[(self.d - self.r):] = 1
        return mask