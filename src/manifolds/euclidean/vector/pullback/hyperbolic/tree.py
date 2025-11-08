from src.diffeomorphisms.vector.r3.tree import TreeVectorDiffeomorphism
from src.manifolds.euclidean.vector.pullback.hyperbolic import HyperbolicdPullbackVectorEuclidean

class TreeHyperbolicdPullbackVectorEuclidean(HyperbolicdPullbackVectorEuclidean):
    
    def __init__(self):
        super().__init__(TreeVectorDiffeomorphism(), needs_embedding=True)