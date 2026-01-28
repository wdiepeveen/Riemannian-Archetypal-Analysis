import torch

from src.diffeomorphisms.vector.product_embedding import ProductEmbeddingVectorDiffeomorphism
from src.diffeomorphisms.vector.poincare import PoincareVectorDiffeomorphism

class PoincareIdentityVectorDiffeomorphism(ProductEmbeddingVectorDiffeomorphism):
    """ Base class describing the identity Diffeomorphism under the Poincaré embedding """

    def __init__(self, d, r):
        assert 0 < r < d, "The Poincaré ball dimension r must be strictly between 0 and the dimension d"
        super().__init__([r, d - r])
        self.r = r
        self.poincare_diffeo = PoincareVectorDiffeomorphism(r)

    def forward(self, x):
        """
        Embed a vector into the product of a Poincaré ball and a Euclidean space
        :param x: N x d
        :return: [N x r, N x d-r]
        """
        return [self.poincare_diffeo.forward(x[:, :self.r]), x[:, self.r:]]
    
    def inverse(self, y):
        """
        Inverse embed a point from the product of a Poincaré ball and a Euclidean space to the vector space
        :param y: [N x r, N x d-r]
        :return: N x d
        """
        return torch.cat([self.poincare_diffeo.inverse(y[0]), y[1]], dim=1)
    
    def differential_forward(self, x, X):
        """

        :param x: N x d
        :param X: N x d
        :return: [N x r, N x d-r]
        """
        return [self.poincare_diffeo.differential_forward(x[:, :self.r], X[:, :self.r]), X[:, self.r:]]
    
    def differential_inverse(self, y, Y):
        """

        :param y: [N x r, N x d-r]
        :param Y: [N x r, N x d-r]
        :return: N x d
        """
        return torch.cat([self.poincare_diffeo.differential_inverse(y[0], Y[0]), Y[1]], dim=1)