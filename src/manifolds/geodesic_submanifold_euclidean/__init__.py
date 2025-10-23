import torch

from src.manifolds import Manifold

class GeodesicSubmanifoldEuclidean(Manifold):
    def __init__(self, d, epoint, ebasis, euclidean_manifold, project_point_iter=1):
        """
        
        :param epoint: [Epoint]
        :param ebasis: d x [Evector]
        :param euclidean_manifold: Euclidean
        """
        super().__init__(d)
        self.ambient_dimension = euclidean_manifold.d
        self.y0 = epoint
        self.V0 = ebasis
        self.euclidean = euclidean_manifold
        self.project_point_iter = project_point_iter

    def l2_project_point(self, x):
        """

        :param x: N x [Epoint]
        :return: N x [Epoint]
        """
        y = self.y0.clone()[None] * torch.ones_like(x) # N x [Epoint]
        print(y.shape, x.shape)
        for _ in range(self.project_point_iter):
            y = self.exp(y, self.l2_project_tangent(y, self.log(y[:,None], x[:,None])[:,0]))[:,0] # N x [Epoint]
        return y
    
    def l2_project_tangent(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Evector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def basis_at_point(self, x):
        """

        :param x: N x [Epoint]
        :return: N x d x [Evector]
        """
        return self.parallel_transport(self.y0[None,None], self.V0[None,None], x[None])[0,0]
    
    def l2_inner(self, x, X, Y):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :param Y: N x L x [Evector]
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def l2_norm(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inner(self, x, X, Y):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :param Y: N x L x [Evector]
        :return: N x M x L
        """
        return self.euclidean.inner(x, X, Y)
    
    def norm(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M
        """
        return self.euclidean.norm(x, X)
        
    def barycentre(self, x):
        """

        :param x: N x [Epoint]
        :return: [Epoint]
        """
        return self.euclidean.barycentre(x)

    def geodesic(self, x, y, t):
        """

        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :param t: K
        :return: N x M x L x K x [Epoint]
        """
        return self.euclidean.geodesic(x, y, t)

    def log(self, x, y):
        """

        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L x [Evector]
        """
        return self.euclidean.log(x, y)

    def exp(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Epoint]
        """
        return self.euclidean.exp(x, X)
    
    def distance(self, x, y):
        """

        :param x: N x M x [Epoint]
        :param y: N x L x [Epoint]
        :return: N x M x L
        """
        return self.euclidean.distance(x, y)

    def parallel_transport(self, x, X, y):
        """

        :param x: N x M x [Epoint]
        :param X: N x M x K x [Evector]
        :param y: N x L x [Epoint]
        :return: N x M x L x K [Evector]
        """
        return self.euclidean.parallel_transport(x, X, y)

