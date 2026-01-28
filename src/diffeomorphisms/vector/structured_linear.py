import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class StructuredLinearVectorDiffeomorphism(VectorDiffeomorphism):
    """ Base class describing a structured linear Diffeomorphism with inverse x \mapsto [[A.T, 0], [B.T, I]] x, where A is r x r, B is r x (d-r) and I is the identity on d-r """

    def __init__(self, W):
        """
            Initialize the structured linear Diffeomorphism
            :param W: r x d matrix defining the structured linear Diffeomorphism (W = [A, B])
        """
        super().__init__(W.shape[1])
        self.r = W.shape[0]
        assert W.shape[1] >= self.r, "Matrix W must have at least r columns"
        self.A = W[:, :self.r]
        self.B = W[:, self.r:]
        self.A_inv = torch.linalg.inv(self.A)

    def forward(self, x):
        """
        Apply the inverse of the structured linear Diffeomorphism to a vector
        :param x: N x d
        :return: N x d
        """
        x1 = x[:, :self.r]
        x2 = x[:, self.r:]
        y1 = x1 @ self.A_inv
        y2 = x2 - y1 @ self.B
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        """
        Apply the structured linear Diffeomorphism to a vector
        :param y: N x d
        :return: N x d
        """
        y1 = y[:, :self.r]
        y2 = y[:, self.r:]
        x1 = y1 @ self.A
        x2 = y2 + y1 @ self.B
        return torch.cat([x1, x2], dim=1)
    
    def differential_forward(self, x, X):
        """
        Apply the differential of the inverse of the structured linear Diffeomorphism to a vector
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        X1 = X[:, :self.r]
        X2 = X[:, self.r:]
        Y1 = X1 @ self.A_inv
        Y2 = X2 - Y1 @ self.B
        return torch.cat([Y1, Y2], dim=1)
    
    def differential_inverse(self, y, Y):
        """
        Apply the differential of the structured linear Diffeomorphism to a vector
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        Y1 = Y[:, :self.r]
        Y2 = Y[:, self.r:]
        X1 = Y1 @ self.A
        X2 = Y2 + Y1 @ self.B
        return torch.cat([X1, X2], dim=1)