import torch

from src.radials.unimodal.elliposoid import EllipsoidUniModalRadial

class MuSPDEllipsoidUniModalRadial(EllipsoidUniModalRadial):
    """ Constructor for the case that the ellipsoid is defined by a covariance matrix Sigma."""
    def __init__(self, mu, Sigma):
        """
        :param mu: d-dimensional tensor representing the center of the ellipsoid.
        :param Sigma: d x d tensor representing the covariance matrix defining the shape of the ellipsoid.
        """
        # compute the semi-major axis, eccentricity, and axis direction from the covariance matrix Sigma
        q = mu / mu.norm()
        a = mu.norm()
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        eps = (1 - eigvals.min() / eigvals.max()).sqrt()

        super().__init__(Sigma.shape[0], a, eps, q)

        eigvals_clamped = torch.clamp(eigvals, min=1e-12)
        sqrt_eigvals = torch.sqrt(eigvals_clamped)

        Sigma_sqrt = (eigvecs * sqrt_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)

        self.Sigma = Sigma
        self.Sigma_sqrt = Sigma_sqrt
        self.Sigma_sqrt_inv = torch.linalg.inv(self.Sigma_sqrt)

    def linear(self, theta):
        """
        :param theta: N x d tensor
        """
        return theta @ self.Sigma_sqrt_inv
