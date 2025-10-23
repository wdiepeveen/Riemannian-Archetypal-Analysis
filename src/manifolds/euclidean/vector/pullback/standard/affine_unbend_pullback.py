import torch

from src.diffeomorphisms.vector.r2.unbend import UnbendVectorDiffeomorphism
from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean

class UnbendStandardPullbackVectorEuclidean(StandardPullbackVectorEuclidean):

    def __init__(self, angle=torch.pi/4, delta=1., eta=0.5):
        super().__init__(UnbendVectorDiffeomorphism(angle, delta, eta))