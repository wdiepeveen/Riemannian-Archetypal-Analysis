from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.image import ImageDiffeomorphism
from src.diffeomorphisms.image.transform import TransformImageDiffeomorphism
        
class NFlowImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width, nflow):
        super().__init__(in_channels, height, width)

        self.nflow = nflow
        self.transform = TransformImageDiffeomorphism(in_channels, height, width, nflow._transform)
        
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
    