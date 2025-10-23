import torch

from src.manifolds.geodesic_submanifold_euclidean import GeodesicSubmanifoldEuclidean

class VectorGeodesicSubmanifoldEuclidean(GeodesicSubmanifoldEuclidean):
    def __init__(self, d, epoint, ebasis, euclidean_manifold, project_point_iter=3):
        super().__init__(d, epoint, ebasis, euclidean_manifold, project_point_iter=project_point_iter)


    def l2_inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)
    
    def l2_norm(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M
        """
        return torch.einsum("NMi,NMi->NM", X, X).sqrt()
    
    def l2_project_tangent(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Evector]
        """
        V = self.basis_at_point(x) # N x d x [Evector]
        G = self.l2_inner(x, V, V) # N x d x d
        VX_inner = self.l2_inner(x, X, V) # N x M x d
        coeffs = torch.linalg.solve(G[:,None], VX_inner) # N x M x d
        PX = (V[:,None,:] * coeffs[:,:,None]).sum(2) # N x M x [Evector]
        return PX