import torch

from src.riemannian_autoencoder.vector import VectorRiemannianAutoencoder

class StandardVectorRiemannianAutoencoder(VectorRiemannianAutoencoder):
    def __init__(self, vector_euclidean, base_point, tangent_basis, ONB=True):
        super().__init__(vector_euclidean)
        """
        Standard Riemannian autoencoder for vector data.
        :param vector_euclidean: VectorEuclidean object
        :param base_point: d Tensor
        :param tangent_basis: r x d Tensor
        """
        self.base_point = base_point
        self.tangent_basis = tangent_basis
        self.onb = ONB

    def encode(self, x):
        """
        Encode data point x onto the tangent space at the base point.
        :param x: N x d Tensor
        :return: N x r Tensor of encoded points
        """
        log_base_point_x = self.manifold.log(self.base_point[None,None], x[None])[0,0]  # N x d
        encoded = torch.einsum('nd,rd->nr', log_base_point_x, self.tangent_basis)  # N x r
        return encoded
    
    def decode(self, z):
        """
        Decode point z from the tangent space at the base point back to the manifold.
        :param z: N x r Tensor of encoded points
        :return: N x d Tensor of decoded points
        """
        if not self.onb:
            g_mat = torch.einsum('rd,sd->rs', self.tangent_basis, self.tangent_basis)  # r x r
            # use torch.lingalg.solve for numerical stability
            z = torch.linalg.solve(g_mat[None], z)  # N x r
        
        log_base_point_x = torch.einsum('nr,rd->nd', z, self.tangent_basis)  # N x d
        decoded = self.manifold.exp(self.base_point[None], log_base_point_x[None])[0]  # N x d
        return decoded