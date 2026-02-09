import torch

from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.inverse_gaussian import InverseGaussianRadial

class CrossRadial(MultiModalRadial):
    def __init__(self):
        super().__init__(2, [InverseGaussianRadial(2, torch.tensor([0.0, 0.0]), torch.tensor([3.0, 0.5])),
                             InverseGaussianRadial(2, torch.tensor([0.0, 0.0]), torch.tensor([0.5, 3.0]))])