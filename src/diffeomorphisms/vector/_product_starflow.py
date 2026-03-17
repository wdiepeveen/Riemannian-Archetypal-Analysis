from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.identity import IdentityDiffeomorphism
from src.diffeomorphisms.vector import VectorDiffeomorphism
from src.diffeomorphisms.vector.product import ProductVectorDiffeomorphism
from src.diffeomorphisms.vector.starflow import StarFlowVectorDiffeomorphism
from src.diffeomorphisms.vector.transform import TransformVectorDiffeomorphism
from src.diffeomorphisms.vector.star import StarVectorDiffeomorphism
from distributions._product import ProductDistribution
from src.distributions.starflows import StarFlowDistribution
from src.distributions.starflows._products.diagonal import StarDiagonalFlowDistribution
        
class ProductStarFlowVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, product_starflow_distribution):
        assert d == product_starflow_distribution.d, "Dimension of diffeomorphism must match dimension of StarFlow distribution"
        super().__init__(d)

        self.starflow = product_starflow_distribution
        self.transform = TransformVectorDiffeomorphism(self.d, self.starflow._transform)
        self.radial = ProductVectorDiffeomorphism([StarVectorDiffeomorphism(self.starflow._distribution.distributions[0].d, self.starflow._distribution.distributions[0]), IdentityDiffeomorphism(self.d - self.starflow._distribution.distributions[0].d)])

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
    