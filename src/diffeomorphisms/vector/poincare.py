import torch
from torch.autograd.functional import jvp

from src.diffeomorphisms.vector import VectorDiffeomorphism
from src.manifolds.hyperbolic.vector.standard import StandardVectorHyperbolic

class PoincareVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d):
        super().__init__(d)
        self.manifold = StandardVectorHyperbolic(d)

    def forward(self, x):
        """
        Embed a vector into the poincare ball
        :param x: N x d
        :return: N x d
        """
        return self.manifold.exp(torch.zeros_like(x), x.unsqueeze(1)).squeeze(1)
    
    def inverse(self, y):
        """
        Inverse embed a point from the poincare ball to the vector space
        :param y: N x d
        :return: N x d
        """
        return self.manifold.log(torch.zeros_like(y), y.unsqueeze(1)).squeeze(1)

    def differential_forward(self, x, X):
        """

        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        _, jvp_result = jvp(lambda p: self.manifold.exp(torch.zeros_like(p), p.unsqueeze(1)).squeeze(1), (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        """

        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        _, jvp_result = jvp(lambda q: self.manifold.log(torch.zeros_like(q.unsqueeze(1)), q.unsqueeze(1)).squeeze(1,2), (y,), (Y,))
        return jvp_result