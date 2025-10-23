from src.diffeomorphisms.vector.r2.banana import BananaVectorDiffeomorphism
from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean

class BananaStandardPullbackVectorEuclidean(StandardPullbackVectorEuclidean):

    def __init__(self, shear=0.1, offset=0.0):
        super().__init__(BananaVectorDiffeomorphism(shear, offset))