import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class CrossVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self):
        super().__init__(3)

    def forward(self, x):
        return 5

    def inverse(self, y):
        return 5