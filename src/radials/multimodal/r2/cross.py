import torch

from src.radials.multimodal import MultiModalRadial

class CrossRadial(MultiModalRadial):
    def __init__(self):
        super().__init__(2)

        self.Sigma_1 = torch.tensor([3.0, 0.5])
        self.Sigma_2 = torch.tensor([0.5, 3.0])
        self.mu_1 = torch.tensor([0.0, 0.0])
        self.mu_2 = torch.tensor([0.0, 0.0])

    def forward(self, theta):
        """
        
        :param theta: N x d tensor with ||theta|| = 1
        :return: N tensor
        """

        rho_1 = 1/torch.norm((theta - self.mu_1[None]) / self.Sigma_1[None], dim=-1) 
        rho_2 = 1/torch.norm((theta - self.mu_2[None]) / self.Sigma_2[None], dim=-1)

        return rho_1 + rho_2