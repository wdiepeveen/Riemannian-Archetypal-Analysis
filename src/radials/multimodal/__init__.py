import torch
import torch.nn as nn
from src.radials import Radial

class MultiModalRadial(Radial):
    def __init__(self, d, radials, aggregation='max'):
        super().__init__(d)
        assert all(radial.d == d for radial in radials), "All radials must have the same dimension as the multimodal radial"
        self.radials = nn.ModuleList(radials)
        self.aggregation = aggregation

    def forward(self, theta):
        """
        
        :param theta: N x d tensor with ||theta|| = 1
        :return: N tensor
        """
        if self.aggregation == 'max':
            return torch.stack([radial(theta) for radial in self.radials], dim=1).max(dim=1).values
        elif self.aggregation == 'softmax':
            radial_values = torch.stack([radial(theta) for radial in self.radials], dim=1)
            weights = torch.softmax(radial_values, dim=1)
            return (weights * radial_values).sum(dim=1)
        elif self.aggregation == 'min':
            return torch.stack([radial(theta) for radial in self.radials], dim=1).min(dim=1).values
        elif self.aggregation == 'softmin':
            radial_values = torch.stack([radial(theta) for radial in self.radials], dim=1)
            weights = torch.softmax(-radial_values, dim=1)
            return (weights * radial_values).sum(dim=1)
        elif self.aggregation == 'sum':
            return sum(radial(theta) for radial in self.radials)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal radial.
        :param idx: int
        :return: UniModalRadial
        """
        return self.radials[idx]