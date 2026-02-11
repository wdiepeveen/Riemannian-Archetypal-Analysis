from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.image import ImageDiffeomorphism
from src.diffeomorphisms.image.to_vec import ToVecImageDiffeomorphism
        
class TransformImageDiffeomorphism(ImageDiffeomorphism):
    """ A vector diffeomorphism implemented using a nflows.transforms object. """
    def __init__(self, in_channels, height, width, transform, vector_input=False, vector_output=False):
        super().__init__(in_channels, height, width)

        self.transform = transform

        self.vector_input = vector_input # whether the transform expects a vector input (N x (C*H*W)) or an image input (N x C x H x W)
        self.vector_output = vector_output # whether the transform outputs a vector (N x (C*H*W)) or an image (N x C x H x W)
        if self.vector_input or vector_output:
            self.to_vec = ToVecImageDiffeomorphism(in_channels, height, width)

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x C x H x W
        :return: N x C x H x W
        """
        out = x.clone()
        if self.vector_input:
            out = self.to_vec(out)
        out, _ = self.transform(out, context=None)
        if self.vector_output:
            out = self.to_vec.inverse(out)
        return out

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x C x H x W
        :return: N x C x H x W
        """
        out = y.clone()
        if self.vector_output:
            out = self.to_vec(out)
        out, _ = self.transform.inverse(out, context=None)
        if self.vector_input:
            out = self.to_vec.inverse(out)
        return out

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        :param x: N x C x H x W
        :param X: N x C x H x W
        :return: N x C x H x W
        """
        _, jvp_result = jvp(lambda x: self.forward(x), (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        :param y: N x C x H x W
        :param Y: N x C x H x W
        :return: N x C x H x W
        """
        _, jvp_result = jvp(lambda y: self.inverse(y), (y,), (Y,))
        return jvp_result
    
    def adjoint_differential_forward(self, x, X):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: N x C x H x W
        :param X: N x C x H x W
        :return: N x C x H x W
        """
        _, vjp_result = vjp(lambda x: self.forward(x), x, X)
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x C x H x W
        :param Y: N x C x H x W
        :return: N x C x H x W
        """
        _, vjp_result = vjp(lambda y: self.inverse(y), y, Y)
        return vjp_result[0]
    
