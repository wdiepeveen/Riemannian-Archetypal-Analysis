import torch

from src.radials.unimodal import UniModalRadial

class EllipsoidRadial(UniModalRadial):
    def __init__(self, d):
        super().__init__(d)

        self.Sigma_inv = self.construct_Sigma_inv()
        
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

    def construct_Sigma_inv(self):
        raise NotImplementedError("construct_Sigma_inv must be implemented by subclasses")
