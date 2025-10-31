import torch
import torch.nn as nn

class RiemannianAutoencoder(nn.Module):
    def __init__(self, euclidean, base_point, r):
        super().__init__()
        self.manifold = euclidean
        self.d = self.manifold.d
        self.base_point = base_point
        self.r = r

    def forward(self, x):
        """
        :param x: N x [Epoint] tensor
        :return : N x [Epoint] tensor
        """
        return self.project_on_manifold(x)

    def encode(self, x):
        """
        :param x: N x [Epoint] tensor
        :return : N x r tensor
        """
        proj_x = self.project_on_manifold(x)  # N x [Epoint]
        log_base_point_x = self.manifold.log(self.base_point[None,None], proj_x[None])[0,0]  # N x [Evector]
        encoded = self.tangent_vector_to_coords(self.base_point[None], log_base_point_x[None])[0]  # N x r
        return encoded

    def decode(self, V):
        """
        :param V: N x r tensor
        :return : N x [Epoint] tensor
        """
        log_base_point_x = self.coords_to_tangent_vector(self.base_point[None], V[None])[0]  # N x [Evector]
        decoded = self.manifold.exp(self.base_point[None], log_base_point_x[None])[0]  # N x d
        return decoded
    
    def tangent_vector_to_coords(self, X):
        """
        :param X: N x M x [Evector] tensor
        :return : N x M x r tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def coords_to_tangent_vector(self, coeffs):
        """
        :param coeffs: N x M x r tensor
        :return : N x M x [Evector] tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def project_on_manifold(self, x):
        """
        :param x: N x [Epoint] tensor
        :return : N x [Epoint] tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def project_on_tangent_space(self, x, X):
        """
        :param x: N x [Epoint] tensor
        :param X: N x M x [Evector] tensor
        :return : N x M x [Evector] tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )