from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.vector import VectorDiffeomorphism
from src.diffeomorphisms.vector.transform import TransformVectorDiffeomorphism
        
class NFlowVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, nflow):
        super().__init__(d)

        self._nflow = nflow
        self.transform = TransformVectorDiffeomorphism(nflow.transform)

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x d
        :return: N x d
        """
        return self.transform(x)

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x d
        :return: N x d
        """
        return self.transform.inverse(y)

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return self.transform.differential_forward(x, X)

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        return self.transform.differential_inverse(y, Y)
    
    def adjoint_differential_forward(self, x, X):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return self.transform.adjoint_differential_forward(x, X)

    def adjoint_differential_inverse(self, y, Y):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        return self.transform.adjoint_differential_inverse(y, Y)
    