import torch.nn as nn
from src.radials import Radial

class MultiModalRadial(Radial):
    def __init__(self, d, radials):
        super().__init__(d)
        assert all(radial.d == d for radial in radials), "All radials must have the same dimension as the multimodal radial"
        self.radials = nn.ModuleList(radials)

    def forward(self, theta):
        """
        
        :param theta: N x d tensor with ||theta|| = 1
        :return: N tensor
        """
        return sum(radial(theta) for radial in self.radials)

    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal radial.
        :param idx: int
        :return: UniModalRadial
        """
        return self.radials[idx]