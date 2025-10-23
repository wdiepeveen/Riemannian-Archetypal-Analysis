import torch

from src.manifolds.geodesic_submanifold_euclidean import GeodesicSubmanifoldEuclidean

class ImageGeodesicSubmanifoldEuclidean(GeodesicSubmanifoldEuclidean):
    def __init__(self, d, epoint, ebasis, euclidean_manifold, project_point_iter=1):
        super().__init__(d, epoint, ebasis, euclidean_manifold, project_point_iter=project_point_iter)


    def l2_inner(self, x, X, Y):
        """

        :param x: N x C x H x W
        :param X: N x M x C x H x W
        :param Y: N x L x C x H x W
        :return: N x M x L
        """
        return torch.einsum("NMchw,NLchw->NML", X, Y)
    
    def l2_norm(self, x, X):
        """

        :param x: N x C x H x W
        :param X: N x M x C x H x W
        :return: N x M
        """
        return torch.einsum("NMchw,NMchw->NM", X, X).sqrt()
    
    def l2_project_tangent(self, x, X):
        """

        :param x: N x C x H x W
        :param X: N x M x C x H x W
        :return: N x M x C x H x W
        """
        V = self.basis_at_point(x) # N x d x C x H x W
        G = self.l2_inner(x, V, V) + 1e-5 * torch.eye(self.d, device=x.device) # N x d x d
        VX_inner = self.l2_inner(x, X, V) # N x M x d
        coeffs = torch.linalg.solve(G[:,None], VX_inner) # N x M x d
        PX = (V[:,None,:,:,:,:] * coeffs[:,:,:,None,None,None]).sum(2) # N x M x C x H x W
        return PX