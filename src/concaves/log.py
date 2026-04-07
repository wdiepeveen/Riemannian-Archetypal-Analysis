import torch

from src.concaves import Concave

class LogConcave(Concave):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    def forward(self, r):
        return torch.log(self.a * r + 1)
    
    def inverse(self, s):
        return (torch.exp(s) - 1) / self.a