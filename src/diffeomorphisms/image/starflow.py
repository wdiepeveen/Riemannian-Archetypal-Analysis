from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.image import ImageDiffeomorphism
from src.diffeomorphisms.image.transform import TransformImageDiffeomorphism
from src.diffeomorphisms.image.star import StarImageDiffeomorphism
        
class StarFlowImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width, starflow):
        super().__init__(in_channels, height, width)

        self.starflow = starflow
        self.transform = TransformImageDiffeomorphism(self.C, self.H, self.W, starflow._transform, vector_output=True)
        self.radial = StarImageDiffeomorphism(self.C, self.H, self.W, starflow._distribution)

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x C x H x W
        :return: N x C x H x W
        """
        return self.radial(self.transform(x))

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x C x H x W
        :return: N x C x H x W
        """
        return self.transform.inverse(self.radial.inverse(y))

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        :param x: N x C x H x W
        :param X: N x C x H x W
        :return: N x C x H x W
        """
        return self.radial.differential_forward(self.transform.forward(x), self.transform.differential_forward(x, X))

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        :param y: N x C x H x W
        :param Y: N x C x H x W
        :return: N x C x H x W
        """
        return self.transform.differential_inverse(self.radial.inverse(y), self.radial.differential_inverse(y, Y))
    