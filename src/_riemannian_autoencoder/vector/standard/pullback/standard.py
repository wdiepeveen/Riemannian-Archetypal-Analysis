from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean
from src._riemannian_autoencoder.vector.standard import StandardVectorRiemannianAutoencoder

class StandardPullbackVectorRiemannianAutoencoder(StandardVectorRiemannianAutoencoder):
    def __init__(self, vector_diffeomorphism, base_point, tangent_basis, ONB=True):
        super().__init__(StandardPullbackVectorEuclidean(vector_diffeomorphism),  base_point, tangent_basis, ONB=ONB)
        self.phi = vector_diffeomorphism
    
