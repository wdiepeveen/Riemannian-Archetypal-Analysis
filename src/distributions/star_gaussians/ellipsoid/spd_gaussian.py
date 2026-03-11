from src.distributions.star_gaussians import StarGaussianDistribution
from src.radials.unimodal.elliposoid.spd import SPDEllipsoidUniModalRadial

class SPDEllipsoidGaussianDistribution(StarGaussianDistribution):
    def __init__(self, Sigma, sign=1):
        super().__init__(Sigma.shape[0], SPDEllipsoidUniModalRadial(Sigma, sign=sign))