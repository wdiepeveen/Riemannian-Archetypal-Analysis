import torch

from src.radials.unimodal.elliposoid.centered import CenteredEllipsoidRadial

class EllipsoidEnclosingCenteredEllipsoidRadial(CenteredEllipsoidRadial):
    def __init__(self, cov):
        super().__init__(cov)

    def construct_Sigma(self):
        return self.cov