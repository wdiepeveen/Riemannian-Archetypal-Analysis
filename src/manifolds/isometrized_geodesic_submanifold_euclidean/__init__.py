from src.manifolds import Manifold
from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean

class l2IsometrizedGeodesicSubmanifoldEuclidean(Manifold):
    def __init__(self, iso_geo_sub_euclidean_manifold):
        super().__init__(iso_geo_sub_euclidean_manifold.d)

        self.euclidean = iso_geo_sub_euclidean_manifold 
        self.ambient_dimension = self.euclidean.euclidean.ambient_dimension
        self.y0 = self.euclidean.euclidean.y0
        self.V0 = self.euclidean.euclidean.V0

    def l2_project_point(self, x):
        """

        :param x: N x [Epoint]
        :return: N x [Epoint]
        """
        return self.euclidean.euclidean.l2_project_point(x)
    
    def l2_project_tangent(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M x [Evector]
        """
        return self.euclidean.euclidean.l2_project_tangent(x, X)

    def l2_inner(self, x, X, Y):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :param Y: N x L x [Evector]
        :return: N x M x L
        """
        return self.euclidean.l2_inner(x, X, Y)
    
    def l2_norm(self, x, X):
        """

        :param x: N x [Epoint]
        :param X: N x M x [Evector]
        :return: N x M
        """
        return self.euclidean.l2_norm(x, X)

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

    def barycentre(self, x, tol=1e-3, max_iter=100, step_size=1/2):
        """

        :param x: N x [Epoint]
        :return: [Epoint]
        """
        return self.euclidean.barycentre(x, tol=tol, max_iter=max_iter, step_size=step_size)

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
    
    def monotonicity_upper_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return self.euclidean.monotonicity_upper_bound(x, X, y, Y)
    
    def lipschitz_lower_bound(self, x, X, y, Y):
        """

        :param x: N x M x d
        :param X: N x M x d
        :param y: N x L x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return self.euclidean.lipschitz_lower_bound(x, X, y, Y)