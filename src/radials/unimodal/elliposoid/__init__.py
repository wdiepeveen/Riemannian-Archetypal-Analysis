import torch

from src.radials.unimodal import UniModalRadial

class EllipsoidRadial(UniModalRadial):
    def __init__(self, d):
        super().__init__(d)

        self.Sigma = self.construct_Sigma()
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        
    def forward(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        return self.compute_intersect(theta)

    def compute_intersect(self, theta):
        """
        :param theta: N x d tensor
        :return: N tensor
        """
        raise NotImplementedError("compute_intersect must be implemented by subclasses")

    def construct_Sigma(self):
        raise NotImplementedError("construct_Sigma must be implemented by subclasses")
