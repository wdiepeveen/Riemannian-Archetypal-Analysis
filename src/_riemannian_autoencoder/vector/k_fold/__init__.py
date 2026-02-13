from src._riemannian_autoencoder.vector import VectorRiemannianAutoencoder

class KfoldVectorRiemannianAutoencoder(VectorRiemannianAutoencoder):
    def __init__(self, vector_euclidean, base_point, tangent_basis, K=5, step_size=0.5, ONB=True):
        super().__init__(vector_euclidean,  base_point, tangent_basis, ONB=ONB)
        self.K = K
        self.step_size = step_size

    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        y = self.base_point.clone()[None].repeat(x.shape[0],1)  # N x d
        for _ in range(self.K):
            y = self.manifold.exp(y, self.project_on_tangent_space(y, self.step_size * (x - y)[:,None]))[:,0]  # N x d
        return y
    
