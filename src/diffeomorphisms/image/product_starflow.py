from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.composition.image import ImageCompositionDiffeomorphism
from src.diffeomorphisms.identity import IdentityDiffeomorphism
from src.diffeomorphisms.image import ImageDiffeomorphism
from src.diffeomorphisms.image.to_vec import ToVecImageDiffeomorphism 
from src.diffeomorphisms.vector.product import ProductVectorDiffeomorphism
from src.diffeomorphisms.vector.starflow import StarFlowVectorDiffeomorphism
from src.diffeomorphisms.image.transform import TransformImageDiffeomorphism
from src.diffeomorphisms.vector.star_gaussian import StarGaussianVectorDiffeomorphism
from src.diffeomorphisms.vector.to_img import ToImgVectorDiffeomorphism
from src.distributions.product import ProductDistribution
from src.distributions.starflows import StarFlowDistribution
from src.distributions.starflows.products.diagonal import StarDiagonalFlowDistribution
        
class ProductStarFlowImageDiffeomorphism(ImageDiffeomorphism):
    def __init__(self, in_channels, height, width, product_starflow_distribution):
        d = in_channels * height * width
        assert d == product_starflow_distribution.d, "Dimension of diffeomorphism must match dimension of StarFlow distribution"
        super().__init__(in_channels, height, width)

        self.starflow = product_starflow_distribution
        self.transform = TransformImageDiffeomorphism(self.C, self.H, self.W, self.starflow._transform, vector_output=True)
        self.radial = ImageCompositionDiffeomorphism(
            [
                ToVecImageDiffeomorphism(self.C, self.H, self.W),
                ProductVectorDiffeomorphism([StarGaussianVectorDiffeomorphism(self.starflow._distribution.distributions[0].d, self.starflow._distribution.distributions[0]), IdentityDiffeomorphism(self.d - self.starflow._distribution.distributions[0].d)]),
                ToImgVectorDiffeomorphism(self.C, self.H, self.W),
            ], self.C, self.H, self.W
                                                     )

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x d
        :return: N x d
        """
        return self.radial(self.transform(x))

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x d
        :return: N x d
        """
        return self.transform.inverse(self.radial.inverse(y))

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return self.radial.differential_forward(self.transform.forward(x), self.transform.differential_forward(x, X))

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        return self.transform.differential_inverse(self.radial.inverse(y), self.radial.differential_inverse(y, Y))
    
    def adjoint_differential_forward(self, x, X):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return self.transform.adjoint_differential_forward(x, self.radial.adjoint_differential_forward(self.transform.forward(x), X))

    def adjoint_differential_inverse(self, y, Y):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        return self.transform.adjoint_differential_inverse(self.radial.inverse(y), self.radial.adjoint_differential_inverse(y, Y))
    