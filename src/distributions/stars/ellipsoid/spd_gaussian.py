from src.distributions.stars import StarDistribution
from src.radials.unimodal.elliposoid.spd import SPDEllipsoidUniModalRadial

class SPDEllipsoidStarDistribution(StarDistribution):
    def __init__(self, Sigma, sign=1):
        super().__init__(Sigma.shape[0], SPDEllipsoidUniModalRadial(Sigma, sign=sign))