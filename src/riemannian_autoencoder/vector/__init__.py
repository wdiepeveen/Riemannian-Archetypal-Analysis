import torch

from src.riemannian_autoencoder import RiemannianAutoencoder

class VectorRiemannianAutoencoder(RiemannianAutoencoder):
    def __init__(self, vector_euclidean,  base_point, tangent_basis, ONB=True):
        super().__init__(vector_euclidean,  base_point, tangent_basis.shape[0])
        self.tangent_basis = tangent_basis
        self.onb = ONB
        if not self.onb:
            # precompute gram matrix and its inverse for efficiency
            self.g_mat = torch.einsum('rd,sd->rs', self.tangent_basis, self.tangent_basis)  # r x r
            self.g_mat_inv = torch.linalg.inv(self.g_mat + 1e-4 * torch.eye(self.r, self.r))  # r x r
    
    def tangent_vector_to_coords(self, X):
        """
        :param X: N x M x d tensor
        :return : N x M x r tensor
        """
        return torch.einsum('nmd,rd->nmr', X, self.tangent_basis)
    
    def coords_to_tangent_vector(self, coeffs):
        """
        :param coeffs: N x M x r tensor
        :return : N x M x d tensor
        """
        if not self.onb:
            coeffs_ = torch.einsum('nmr,rs->nms', coeffs, self.g_mat_inv)  # N x M x r
        else:
            coeffs_ = coeffs.clone()
        
        return torch.einsum('nr,rd->nd', coeffs_, self.tangent_basis)  # N x d
    
    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def project_on_tangent_space(self, x, X):
        """
        :param x: N x d tensor
        :param X: N x M x d tensor
        :return : N x M x d tensor
        """
        tangent_basis_x = self.manifold.parallel_transport(self.base_point[None,None], self.tangent_basis[None,None], x[None])[0,0]  # N x r x d
        g_mat_x = torch.einsum('nrd,nsd->nrs', tangent_basis_x, tangent_basis_x)  # N x r x r
        
        coeffs = torch.einsum('nmd,nrd->nmr', X, tangent_basis_x)  # N x M x r
        coeffs = torch.linalg.solve(g_mat_x + 1e-4 * torch.eye(self.r, self.r)[None], coeffs.permute(0,2,1)).permute(0,2,1)  # N x M x r
        
        return torch.einsum('nmr,nrd->nmd', coeffs, tangent_basis_x)  # N x M x d