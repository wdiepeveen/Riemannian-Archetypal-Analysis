from src.distributions.stars import StarDistribution
from src.radials.multimodal.inverse_affine_norm.diagonal import MultiInverseDiagonalNormRadial

class MultiInverseDiagonalNormStarDistribution(StarDistribution):
    def __init__(self, d, num_radials, sigma=1e-6):
        super().__init__(d, MultiInverseDiagonalNormRadial(d, num_radials, sigma=sigma))
        
    def extract_mode(self, idx):
        """
        Extract the idx-th mode of the multimodal star Gaussian distribution.
        :param idx: int
        :return: StarDistribution with the same radial as the idx-th mode of the multimodal radial
        """
        return StarDistribution(self.d, self.radial.extract_mode(idx))