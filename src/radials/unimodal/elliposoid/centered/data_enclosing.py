import torch

from src.radials.unimodal.elliposoid.centered import CenteredEllipsoidRadial

class DataEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, data, reg_param=1e-2):
        self.m = data
        self.reg_param = reg_param

        super().__init__(data.shape[1])

    def construct_Sigma_inv(self):
        _U, _, _ = torch.linalg.svd(self.m.T)
        r = min(self.m.shape[0], self.d)
        U = _U[:, :r]
        sq_norms = (self.m @ U) ** 2
        lambda_ = r * torch.max(sq_norms, dim=0).values
        Sigma_inv = 1 / self.reg_param * (torch.eye(self.d) - U @ U.T) + U @ torch.diag(1/(lambda_ + self.reg_param)) @ U.T
        return Sigma_inv