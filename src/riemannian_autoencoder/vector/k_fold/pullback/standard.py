from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean
from src.riemannian_autoencoder.vector.k_fold import KfoldVectorRiemannianAutoencoder

class StandardPullbackKfoldVectorRiemannianAutoencoder(KfoldVectorRiemannianAutoencoder):
    def __init__(self, vector_diffeomorphism, base_point, tangent_basis, K=5, step_size=0.5, ONB=True):
        super().__init__(StandardPullbackVectorEuclidean(vector_diffeomorphism),  base_point, tangent_basis, K=K, step_size=step_size, ONB=ONB)
        self.phi = vector_diffeomorphism
    
