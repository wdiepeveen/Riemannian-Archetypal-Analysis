import torch

from src.radials.unimodal.centered_ellipsoid import CenteredEllipsoidRadial

class EllipsoidEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, cov):
        self.cov = cov
        super().__init__(cov.shape[0])

    def construct_Sigma_inv(self):
        return torch.linalg.inv(self.cov)