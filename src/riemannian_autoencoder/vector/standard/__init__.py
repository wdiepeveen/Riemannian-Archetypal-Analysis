import torch

from src.riemannian_autoencoder.vector import VectorRiemannianAutoencoder

class StandardVectorRiemannianAutoencoder(VectorRiemannianAutoencoder):
    def __init__(self, vector_euclidean, base_point, tangent_basis, ONB=True):
        super().__init__(vector_euclidean,  base_point, tangent_basis, ONB=ONB)

    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        return self.manifold.exp(self.base_point[None], self.project_on_tangent_space(self.base_point[None], self.manifold.log(self.base_point[None,None], x[None])[0,0][None]))[0]
    