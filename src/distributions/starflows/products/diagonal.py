from src.distributions.diagonal_gaussian import DiagonalGaussian
from src.distributions.starflows.products import ProductStarFlowDistribution

class StarDiagonalFlowDistribution(ProductStarFlowDistribution):
    def __init__(self, d, transform, star_gaussian_distribution):
        assert d > star_gaussian_distribution.d, "Dimension of product StarFlow distribution must be larger than dimension of star Gaussian component"
        super(StarDiagonalFlowDistribution, self).__init__(d, transform, star_gaussian_distribution, DiagonalGaussian(d - star_gaussian_distribution.d))