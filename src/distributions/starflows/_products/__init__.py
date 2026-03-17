from distributions._product import ProductDistribution
from src.distributions.starflows import StarFlowDistribution

class ProductStarFlowDistribution(StarFlowDistribution):
    def __init__(self, d, transform, star_gaussian_distribution, diagonal_gaussian_distribution):
        assert d == star_gaussian_distribution.d + diagonal_gaussian_distribution.d, "Dimension of product StarFlow distribution must match sum of dimensions of components"
        product_distribution = ProductDistribution(d, [star_gaussian_distribution, diagonal_gaussian_distribution])
        super(ProductStarFlowDistribution, self).__init__(d, transform, product_distribution)