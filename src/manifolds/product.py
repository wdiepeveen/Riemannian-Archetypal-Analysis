import torch

from src.manifolds.euclidean.vector import VectorEuclidean

class ProductVectorEuclidean(VectorEuclidean):
    """ Base class describing a product manifold M_1 x M_2 x ... x M_m """

    def __init__(self, manifolds):
        super().__init__(sum([manifold.d for manifold in manifolds]))
        self.ds = [manifold.d for manifold in manifolds]
        self.manifolds = manifolds
        self.m = len(self.ds)

    def barycentre(self, x, tol=None, max_iter=None, step_size=None, red_coef=None):
        """

        :param x: [N x Mpoint_1, ..., N x Mpoint_m]
        :return: [Mpoint_1, ..., Mpoint_m]
        """
        barys = []
        for i in range(self.m):
            barys.append(self.manifolds[i].barycentre(x[i], tol=tol, max_iter=max_iter, step_size=step_size, red_coef=red_coef))
        return barys

    def inner(self, x, X, Y):
        """

        :param x: [N x Mpoint_1, ..., N x Mpoint_m]
        :param X: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :param Y: [N x L x Mpoint_1, ..., N x L x Mpoint_m]
        :return: N x M x L
        """
        inner_prods = 0
        for i in range(self.m):
            inner_prods = inner_prods + self.manifolds[i].inner(x[i], X[i], Y[i])
        return inner_prods
        
    def norm(self, x, X):
        """

        :param x: [N x Mpoint_1, ..., N x Mpoint_m]
        :param X: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :return: N x M
        """
        norms = 0
        for i in range(self.m):
            norms = norms + self.manifolds[i].norm(x[i], X[i])**2
        return torch.sqrt(norms)
        
    def geodesic(self, x, y, t):
        """

        :param x: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :param y: [N x L x Mpoint_1, ..., N x L x Mpoint_m]
        :param t: K or N x M x L x K
        :return: [N x M x L x K x Mpoint_1, ..., N x M x L x K x Mpoint_m]
        """
        geodesics = []
        for i in range(self.m):
            geodesics.append(self.manifolds[i].geodesic(x[i], y[i], t))
        return geodesics
            
    def log(self, x, y): 
        """

        :param x: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :param y: [N x L x Mpoint_1, ..., N x L x Mpoint_m]
        :return: [N x M x L x Mpoint_1, ..., N x M x L x Mpoint_m]
        """
        logs = []
        for i in range(self.m):
            logs.append(self.manifolds[i].log(x[i], y[i]))
        return logs

    def exp(self, x, X): 
        """

        :param x: [N x Mpoint_1, ..., N x Mpoint_m]
        :param X: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :return: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        """
        exps = []
        for i in range(self.m):
            exps.append(self.manifolds[i].exp(x[i], X[i]))
        return exps

    def distance(self, x, y):
        """

        :param x: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :param y: [N x L x Mpoint_1, ..., N x L x Mpoint_m]
        :return: N x M x L
        """
        dists = 0
        for i in range(self.m):
            dists = dists + self.manifolds[i].distance(x[i], y[i])**2
        return torch.sqrt(dists)

    def parallel_transport(self, x, X, y):
        """

        :param x: [N x M x Mpoint_1, ..., N x M x Mpoint_m]
        :param X: [N x M x K x Mpoint_1, ..., N x M x K x Mpoint_m]
        :param y: [N x L x Mpoint_1, ..., N x L x Mpoint_m]
        :return: [N x M x L x K x Mpoint_1, ..., N x M x L x K x Mpoint_m]
        """
        pts = []
        for i in range(self.m):
            pts.append(self.manifolds[i].parallel_transport(x[i], X[i], y[i]))
        return pts
        