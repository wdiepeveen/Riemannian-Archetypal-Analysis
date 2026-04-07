import torch

from src.radials.multimodal import MultiModalRadial
from src.radials.unimodal.elliposoid.centered.ellipsoid_enclosing import EllipsoidEnclosingCenteredEllipsoidRadial

class CrossRadial(MultiModalRadial):
    def __init__(self):
        super().__init__(2, [EllipsoidEnclosingCenteredEllipsoidRadial(torch.tensor([[3.0, 0.0], [0.0, 0.1]])),
                             EllipsoidEnclosingCenteredEllipsoidRadial(torch.tensor([[0.1, 0.0], [0.0, 3.0]]))])