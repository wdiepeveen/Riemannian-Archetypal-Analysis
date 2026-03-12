from src.distributions.stars import StarDistribution
from src.radials.multimodal.ellipsoid.spd import MultiSPDEllipsoidRadial

class MultiSPDEllipsoidStarDistribution(StarDistribution):
    def __init__(self, d, Sigmas, signs, aggregation='sum'):
        super().__init__(d, MultiSPDEllipsoidRadial(d, Sigmas, signs, aggregation=aggregation))
        
    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal star Gaussian distribution.
        :param idx: int
        :return: StarDistribution with the same radial as the idx-th mode of the multimodal radial
        """
        return StarDistribution(self.d, self.radial.extract_mode(idx))